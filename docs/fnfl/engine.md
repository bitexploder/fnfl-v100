# How the FNFL engine works

FNFL's engine runs one model, Qwen3.8-Flash-Next, for one user on two Tesla V100 32 GB cards. It is host C++ and CUDA in [`fn/`](../../fn/), written for this model and this hardware.

- No ggml or llama.cpp GPU kernel runs in it.
- llama.cpp contributes the HTTP server, the tokenizer, the chat templates and a GGUF reader.
- The server reaches the engine through [`fn/fnfl.h`](../../fn/fnfl.h), which has five working calls:
  - `fnfl_load`
  - `fnfl_set_sampling`, for a request's options
  - `fnfl_prefill`, for new prompt tokens
  - `fnfl_step`, which returns the next 1 to 5 tokens
  - `fnfl_truncate`

This page describes the model, how it is split over the two cards, what a decode step and a prompt chunk do, the engine's memory and numerics, and the measured speed. The source files named here carry the full contracts in their header comments.

## The model

Qwen3.8-Flash-Next (GGUF architecture `qwen4exp`) is a 48-layer mixture-of-experts model. The engine refuses any file whose shapes differ from these ([`fn/model/shapes.h`](../../fn/model/shapes.h)):

- **Width:** 2,560.
- **Vocabulary:** 248,320 tokens.
- **Trained context:** 262,144 tokens.
- **Hyper-connections (HC):** the residual is four parallel streams of 2,560 values rather than one.
  - Before each sublayer, a small low-rank mixer (rank 320) combines the four streams into that sublayer's input.
  - After the sublayer, its output is written back into each stream with learned weights.
  - A final mixer combines the streams before the output head.
- **Token mixing,** in a repeating pattern of three GDN layers and one QSA layer:
  - **36 GDN layers** (gated delta-net linear attention). Each has 48 value heads of width 128 and a short causal convolution of width 4. Each head carries a 128 × 128 state that the gated delta rule updates once per token, so the cost per token does not grow with the history.
  - **12 QSA layers** (layers 3, 7, …, 47), which are sparse full attention.
    - There are 24 query heads and 2 key/value heads of width 256, with a sigmoid output gate.
    - A small indexer scores the history in blocks of four cells. Each query then attends only to the 512 highest-scoring blocks and the incomplete last block, at most 2,051 cells whatever the depth.
- **Feed-forward:** 512 routed experts, of which each token uses 10, each 640 wide. A shared expert, also 640 wide, runs for every token.
- **Per-layer embedding (PLE):** besides the normal token embedding, each token reads 16 rows of a large n-gram table.
  - The rows are chosen by hashing the token together with the one or two tokens before it.
  - They are decoded into a 2,560-value input for a PLE layer that runs before layer 1.
  - The table is 28.8 GB (26.9 GiB) in IQ4_NL and stays in host memory.
- **Multi-token prediction (MTP) head:** one extra block with its own attention and experts, distributed separately as a Q8_0 GGUF. It predicts the next token from the target's last hidden state and the current token. FNFL uses it as the draft model.

**The quantized file is mixed.**

- The routed experts' gate and up weights use IQ3_S (13 layers), IQ2_XS (10), IQ2_S (10), IQ2_XXS (9) or IQ3_XXS (6).
- Their down weights use Q2_0 (30 layers) or IQ4_NL (18).
- The shared expert uses IQ3_S to Q6_K, depending on the layer.
- The dense projections are mostly Q6_K, Q5_K, IQ4_XS and Q4_K, with some BF16, IQ3_S and Q8_0.
- The output head is Q5_K.
- The HC weights are BF16.

## The hardware it is built for

- **The cards:** two Tesla PG500-216 cards, each a GV100 with 72 SMs (not the usual 80), 32 GB of HBM2 and 6 MiB of L2.
  - They hold 1,380 MHz under sustained load.
  - Measured DRAM reads reach about 1,020 GB/s per card.
- **The link:** the cards sit on PCIe 3.0 x8 each with no NVLink, so card-to-card messages cross the host bridge.
- **What Volta lacks:** `cp.async`, TMA, asynchronous barriers and BF16 arithmetic. Its tensor cores take FP16 only.

Generating text for one user reads every dense weight and the selected experts' weights at each step, through hundreds of small kernels and about a hundred card-to-card hand-offs. Most of the engine's design follows from three aims:
- keeping those reads near the memory's speed;
- keeping the hand-offs short;
- keeping both cards busy at once.

## Splitting the model over two cards

FNFL splits every layer in half rather than giving each card half the layers. Both cards work on every token at the same time, and they exchange partial results after each sublayer. The placement ([`fn/model/placement.h`](../../fn/model/placement.h), [`fn/model/layer.h`](../../fn/model/layer.h)) gives card *r* (0 or 1):

- **HC:** features 1,280 × *r* to 1,280 × *r* + 1,279 of each of the four streams.
- **GDN:** 24 of the 48 value heads and their query/key groups, with 5,120 convolution channels.
- **QSA:** 12 of the 24 query heads and one of the two key/value heads, so each card keeps the keys and values of its own head.
  - The indexer of each QSA layer runs on one card, alternating by layer.
  - It sends its block selection to the other card.
- **Experts:** channels 320 × *r* to 320 × *r* + 319 of every routed expert and of the shared expert.
  - Each card computes its half of all ten selected experts.
  - The router runs on both cards, so the expert choice needs no exchange.
- **Output head:** token IDs 0–124,159 on card 0 and 124,160–248,319 on card 1. The draft head's 65,536-token subset is split in half between the cards in the same way.

**Exchanges** ([`fn/xchg/`](../../fn/xchg/)) are direct peer-to-peer writes into the other card's memory. Every sublayer ends with one, which makes about a hundred exchanges per decode pass, so their latency is a large part of the design.

**Sampling** happens on both cards from the same exchanged values, with the same machine code. The two cards reach the same decision without one card broadcasting it.

## A decode step

[`fn/runtime/runtime.h`](../../fn/runtime/runtime.h) runs the speculative loop. Each `fnfl_step` call returns the tokens of one step.

1. **Bootstrap.** After a prompt, the first token is sampled from the logits the prompt's last row left behind.
2. **Draft.** The MTP head runs *d* times in a chain (`--fnfl-spec-d`, 1 to 4). Each run takes the previous token and hidden state and proposes the next token from its 65,536-token subset.
   - At temperature 0 the proposal is the draft's argmax. Above 0 it is sampled with top-k, and the draft's probabilities are kept for the accept step.
   - The draft reads the conversation through its own attention. With `dense` it reads every cell. With `sparse` it reads the 512 blocks that the target's indexer selected.
3. **Verify.** The target model runs once over *d* + 1 rows: the last committed token followed by the *d* proposals. This produces the target's distribution at each position. The cost of a pass grows only slowly with its rows, because the weights are read once for all of them.
4. **Accept.** Proposals are checked in order.
   - At temperature 0, a proposal is accepted when it equals the target's argmax.
   - Above 0, proposal *x* is accepted with probability min(1, p(x)/q(x)), where *p* is the target's probability and *q* the draft's. On the first rejection, a replacement is drawn from the normalized residual max(0, p − q).
   - If every proposal is accepted, one more token is drawn from the last row.

   This is standard speculative sampling, and it leaves the output distributed exactly as the target model's own. A step returns from one to *d* + 1 tokens.
5. **Commit.** The verify pass wrote state for all of its rows.
   - Recurrent GDN state is kept in versions: the committed one and one after each row of the step. Committing the accepted rows moves a single index to the right version; nothing is copied back.
   - Attention keys and values of rejected rows are simply overwritten later.
   - An end-of-generation token ends the step there.
6. **Catch-up.** The draft processes the committed tokens to bring its own keys, values and hidden carry up to date for the next step.

A speculative step has three points where the host waits for the cards. Everything between them is replayed from CUDA graphs captured at load, one per card for each row count. A small per-step descriptor in device memory carries the depth, positions and tokens, so one graph serves every depth.

**Where the time goes:** this is a step at 130,865 tokens of history with four sparse proposals (2026-10-16):

- the draft's four chained steps took 1.74 ms;
- the five-row verify took 14.2 ms;
- the catch-up took 0.2 ms;
- the whole step took 16.2 ms.

That run averaged 3.13 tokens a step, for 194.8 tokens/s. The step time barely moves with depth, because QSA reads at most 2,051 cells a query and GDN reads none. Speed depends mostly on how many proposals the target accepts.

## Prompt processing

[`fn/runtime/prefill.h`](../../fn/runtime/prefill.h) and [`fn/model/prompt.h`](../../fn/model/prompt.h) handle prompts.

- **Chunks:** a prompt is processed in chunks of up to 8,192 tokens that never cross a multiple of 8,192.
- **Halves:** each chunk runs as two halves of up to 4,096 rows, layer by layer. While half A's attention result crosses to the other card, half B computes, and so on through the FFN, so exchanges overlap compute. These large messages go through the cards' copy engines as FP16 partial results.
- **Tensor cores:** the dense products, the HC mixers and the router run on the tensor cores (FP16 operands, F32 accumulation; [`fn/kernels/tc/tc_gemm.cuh`](../../fn/kernels/tc/tc_gemm.cuh)).
- **Experts:** prompt rows are sorted by expert, and each expert's weights are unpacked to FP16 once for all the rows that chose it.
- **GDN** runs the gated delta rule in its chunked form, 64 rows at a time ([`fn/kernels/gdn/gdn_prompt.h`](../../fn/kernels/gdn/gdn_prompt.h)).
- **HC replicas:** during a prompt, each card holds both halves of the HC weights, so the mixers need no exchange.
- **Draft:** the draft catches up inside each chunk, so decoding can start as soon as the last chunk ends.
- **PLE:** the next chunk's PLE rows are gathered from the host table on a CPU thread while the current chunk computes. The first chunk's gather cannot overlap, which is most of the fixed cost of a short append.
- **Checkpoints:** at every multiple of 8,192 tokens and at the end of each prompt, the engine copies the recurrent state, so a later prompt that diverges can resume from it.
  - This covers the GDN states and convolution histories, the PLE history and the draft's carry. Keys and values are not copied, since restoring just shortens their valid length.
  - Up to eight checkpoints are kept.
- **Failures:** a chunk that fails publishes nothing, so the slot's state stays consistent.

## Memory

Each card reserves everything at load, sized for the `-c` context, and allocates nothing while serving. [`fn/model/state.h`](../../fn/model/state.h) and `state.cpp` list every buffer with the formula for its size.

- **Weights:** 23.2 GiB per card for the target model, plus the draft model's half and the draft head's re-sharded copy of the output head.
- **Per context token:** about 16 KB per card.
  - The target's QSA keys and values are BF16, 12 KB.
  - The draft's keys and values are F16, 1 KB.
  - The indexer's keys and block summaries account for most of the rest.
- **Fixed state:**
  - GDN states in six versions (F32, 128 × 128 per head)
  - the prompt-chunk working space
  - the CUDA graphs
  - the exchange rings
- **Free:** at 198,720 cells, 2.41 GiB is left free per card; at 262,144, 1.47 GiB.

The host keeps the two model files memory-mapped. After the load, only the PLE table is read, 16 small rows per token. `--fnfl-ple-table full` pins it in RAM; `lazy` relies on the page cache and prefetch hints.

## Numerics

The engine's numerical contract sets out what counts as correct.

- **The reference is the model's math as the pinned llama.cpp release (`66e11c3`) computes it,** not its exact arithmetic. FNFL is correct when its outputs differ from that reference by no more than twice the reference's own spread between its prompt and decode modes.
- **Every decode kernel is deterministic and batch-invariant.** A row's results are bit-for-bit the same whether it is computed alone or as one of five rows in a verify pass. That is why greedy speculative decoding produces exactly the tokens plain greedy decoding would.
- **Prompt kernels are deterministic but not batch-invariant.** Their FP16 tensor-core products differ slightly from the decode path, as llama.cpp's own prompt and decode modes do.
- **Decode products** take activations as int8 codes with scales and use `IDP4A` dot products on the packed weight codes. GDN state and accumulations are F32.

## Sampling

The sampler ([`fn/kernels/samp/samp.h`](../../fn/kernels/samp/samp.h)) follows llama.cpp's chain order.

- Top-k, top-p and min-p choose the kept tokens from the untempered logits, and temperature then shapes their probabilities.
- Each card takes its own top *k*, the two lists are exchanged and merged, and both cards draw the same token from the same seeded random stream.
- The served paths are greedy (temperature 0) and top-k from 1 to 256.
- The kernels also apply logit bias and the repetition, frequency and presence penalties, but the server does not pass them through and refuses them per request.
- DRY, XTC, typical-p, mirostat and grammars are not in the sampler at all, and the server refuses them too (see [server.md](server.md#sampling-per-request)).

## Performance

These figures were measured on the development machine, two PG500-216 cards at 1,380 MHz with driver 580 and CUDA 12.8, one request at a time.

- **Fresh prompt:** 130,865 tokens in 36.78 s, 3,558 tokens/s.
- **Appending to a deep conversation:** 3,391 new tokens after restoring 196,608, in 1.281 s from request to decode-ready, 2,647 tokens/s. The first chunk's host PLE gather took 128 ms of that.
- **Decode at 130,865 tokens of history, real text:**
  - With four proposals and sparse draft attention, 194.8 tokens/s over a 189-token reply, at 3.13 tokens and 16.2 ms a step.
  - With two proposals and dense draft attention, the earlier serving configuration, about 190 tokens/s.
  - The two runs used different sampling settings and are not a matched comparison.
- **Decode at about 200K:** 220 tokens/s on a 125-token reply with two proposals, at 12.7 ms a step.
- **In use:** the server log of an agentic coding run (four proposals, sparse draft attention, 262,144 cells, 2026-10-15) gives these figures.
  - 37 replies of 2,000 tokens or more, on histories from 15K to 150K, ran at 178–289 tokens/s with a median of 209.
  - The median step time was 16.1–16.3 ms throughout.
  - These are unmatched requests, not a controlled measurement.
- **Agentic tasks:** four coding tasks driven through the server passed 53 of 53 checks. This supports those workflows; it is not a general quality score.

The project's targets are:
- decode: 200 tokens/s at depth, as a floor;
- fresh prompts: 4,000 tokens/s up to 192K;
- appends: 3,000 tokens/s for 4K appends at 128K–192K.

Decode at depth is close to the floor, at 194.8 tokens/s in the run at 130K and a median of 209 in use. Prompt processing is at 89% of its fresh-prompt target and 88% of its append target.

## Source map

- **`fn/fnfl.h`:** the server's interface to the engine.
- **`fn/glue/`:** implements that interface. It checks the file and the launch, then loads.
- **`fn/runtime/`:** the request flow:
  - the speculative loop (`runtime`)
  - the target's round frame and versions (`target`)
  - prompt chunking and checkpoints (`prefill`)
  - the host PLE gather (`ple_host`)
  - the developer driver's commands
- **`fn/model/`:**
  - the model's shapes
  - the placement over two cards
  - one layer, the target pass, the draft block and the prompt path
  - the reserved state
- **`fn/kernels/`:** the CUDA kernels, by family:
  - `hc` (hyper-connections)
  - `gdn`
  - `qsa` (the indexer and attention)
  - `experts`
  - `dense` (quantized products)
  - `tc` (the tensor-core core)
  - `act` (the activation format)
  - `embed`
  - `ple`
  - `head` (the output head)
  - `samp` (the sampler)
  - `draft` (the MTP block's attention)
  - `round` (the per-step descriptor and versions)
- **`fn/xchg/`:** card-to-card exchanges for decode and prompts.
- **`fn/loader/`:** reads the GGUF, places each tensor slice on its card and reserves the state.
- **`fn/quant/`:** CPU decoders and synthetic blocks for every weight format, used by the tests.
- **`fn/borrowed/`:** the only route into llama.cpp's CPU code, for GGUF metadata and tensor directories.
- **`fn/tools/`:**
  - the developer driver `fnfl-drive`
  - the include checker
  - the launch script
- **`fn/proto/`:** prototypes kept for their tests.

## Known limits

- **One model file, one draft file, two 32 GB V100s as devices 0 and 1, one conversation at a time.**
- **No constrained output.** Requests with a grammar or a JSON schema are refused, as are the other unsupported sampling options.
- **Paths from the development machine** are built into:
  - the preset's compilers and web-UI archive;
  - the default draft subset (always pass `--fnfl-draft-subset`);
  - `fn/tools/fnfl-serve.sh`;
  - `fnfl-drive`'s model paths.
- **No context shift:** a conversation ends at its context size.
- **Prefix reuse only within the last eight checkpoints**, plus pure extensions. A prompt that diverges further back is read again from the start.
