# Running the FNFL server

FNFL's server is llama.cpp's `llama-server` with the engine swapped out. The HTTP layer, the tokenizer, chat templates, tool-call parsing, stop strings and streaming are llama.cpp's. The model is loaded and run by FNFL's engine (`fn/`) through the five calls in [`fn/fnfl.h`](../../fn/fnfl.h). This page covers starting it, the arguments that matter, the API, and reading its log.

Building it is in [build.md](build.md). How the engine works is in [engine.md](engine.md).

## Starting it

```sh
build-fnfl/bin/llama-server --host 127.0.0.1 --port 8190 -np 1 --jinja \
-m /path/to/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
-md /path/to/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
--fnfl-draft-subset models/draft-vocab-qwen3.8-flash-next.txt \
-c 262144 --fnfl-spec-d 4 --fnfl-draft-attention sparse
```

The last line gives the model's full context and the settings the project serves with: four proposals a step and sparse draft attention. Without the two FNFL settings, the server proposes two drafts a step with dense draft attention.

The server uses CUDA devices 0 and 1 and needs both. To run it on two other cards, set `CUDA_VISIBLE_DEVICES` so that they appear as 0 and 1. Nothing else may be using them, since the engine reserves nearly all of each card's memory.

Loading takes about 8 s when the model files are already in the page cache, and longer when they have to come off disk. During the load, `/health` answers 503. When it finishes, the log prints these lines:

```
FNFL: model loaded in 7.75 s (open 0.38, fill 7.35, reserve 0.02) at 262144 cells; free 1.465 / 1.467 GiB
FNFL: --fnfl-ple-table lazy (default)
FNFL: PLE table: lazy, not pinned (the plain mapping; PL1's hints before each gather)
FNFL settings: slots 1; cells 262144; draft on, attention sparse (SP2's switch), d 4, subset ...; end-of-generation IDs [248044,248046,248063,248064,248065]; launch sampler T 1.00 top-k 20 top-p 0.95 min-p 0.05 ...
... load_model: initializing, n_slots = 1, n_ctx_slot = 262144
... llama_server: listening on http://127.0.0.1:8190
```

- **`free a / b GiB`** is what is left on cards 0 and 1 after the engine's reservation. It should be above zero on both.
- **`FNFL settings`** echoes every engine setting in force: the cells, the draft attention, the proposal count `d`, the subset file, the end-of-generation tokens and the launch sampling. Check this line first when something behaves unexpectedly.

To stop the server, send it SIGTERM. Both cards should then return to 0 MiB.

## Arguments

### Required

- **`-m FILE`:** the first shard of the target GGUF. The second shard is found from the first's name, so keep both in the same directory.
  - The engine reads the file's metadata and refuses it unless the architecture is `qwen4exp` and every shape matches Qwen3.8-Flash-Next ([`fn/model/shapes.h`](../../fn/model/shapes.h)).
  - It also expects the tensor formats of the ISTA-DASLab GSQ-RCO IQ3_XXS file. A different quantization of the same model fails the load with a message naming the first tensor that differs.
- **`-md FILE`:** the MTP draft GGUF, `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`. Without it the load stops with "FNFL needs the MTP draft (-md)". There is no decode path without the draft.
- **`-np 1`:** one slot. Any other value stops the load with "FNFL serves one slot". Leaving `-np` out also fails, since the server's default is automatic (4).
- **`--fnfl-draft-subset FILE`:** the draft head's vocabulary. This is a list of exactly 65,536 distinct token IDs, separated by whitespace (the shipped file has one per line).
  - The draft proposes tokens only from this list. The target model still verifies against its full 248,320-token vocabulary, so the subset affects how often drafts are accepted, not what can be generated.
  - Use `models/draft-vocab-qwen3.8-flash-next.txt` from this repository.
  - **Always pass it.** Its built-in default is a path on the development machine ([`fn/glue/fnfl_glue.cpp`](../../fn/glue/fnfl_glue.cpp)), so leaving it out fails the load anywhere else.

### Strongly recommended

- **`--jinja`:** renders chats with the model's own Jinja template from the GGUF. That template handles thinking and tool calls; without `--jinja` the server falls back to a generic template.
- **`-c N`:** the context, in tokens (the engine calls them cells).
  - The slot gets `min(N, 262,144)`. `-c 0` or no `-c` gives the full 262,144.
  - The engine reserves the whole context at load. More context leaves less free memory on each card: 198,720 cells leave 2.41 GiB free per card, and 262,144 leave 1.47 GiB.
  - When a conversation reaches the end of its context, generation stops with the length reason. There is no context shift, and a prompt longer than the context is refused.
- **`--host` and `--port`:** as in llama.cpp. The default is `127.0.0.1:8080`. Use `--api-key` if the port is reachable by others.

### Launch sampling

- `--temp T` (default 1.0), `--top-k K` (default 20), `--top-p P` (default 0.95) and `--min-p P` (default 0.05) set the sampling a request gets when it does not choose its own. The defaults are the model's recommended settings.
- At a temperature above 0, top-k must be from 1 to 256.

### FNFL settings

- **`--fnfl-spec-d N`** (1 to 4, default 2): how many draft tokens each speculative step proposes.
  - More proposals pay off when the text is predictable, as code and tool calls tend to be.
  - The step that verifies them is dearer. At 130K–200K of history, a step takes about 16.2 ms with four proposals, against about 12.7 ms with two.
  - The development machine serves with 4.
- **`--fnfl-draft-attention dense|sparse`** (default `dense`): how the draft attends to the conversation.
  - `dense` reads every cell.
  - `sparse` reads only the 512 blocks the target's own sparse-attention indexer picked, which is much cheaper at depth.
  - The development machine serves with `sparse`, together with `--fnfl-spec-d 4`, and `fnfl-serve.sh` starts with both by default.
- **`--fnfl-ple-table lazy|full`** (default `lazy`): what happens to the model's 26.9 GiB per-layer-embedding table.
  - `lazy` leaves it memory-mapped and hints the pages each step will read, so the first requests may fault some in from disk.
  - `full` reads the whole table into RAM at startup and pins it there. That makes startup longer and needs about 27 GiB of free host memory.
  - Either way the results are the same; only timing differs.

### Test switches

These exist for the engine's own checks and are not for serving.

- **`--fnfl-check-state`:** after every engine call, compares the engine's tokens with the server's and fails the request on a mismatch. It also turns on a per-step log line.
- **`--fnfl-test-eog ID`:** adds a token to the end-of-generation set. It can be repeated.
- **`--fnfl-test-fail-step K`:** makes step K of the next request fail once, to exercise error handling.

### Refused at launch

The server stops at load with a message for:

- `--embedding` or `--rerank`
- `--mmproj` (multimodal input)
- `--lora`

### Accepted but without effect

llama.cpp's options for its own backend are parsed and then ignored, because FNFL places and runs everything itself. These include:

- `-ngl`, `--split-mode`, `--tensor-split` and `--main-gpu`
- `-fa` and the KV cache types `-ctk` and `-ctv`
- `-b`, `-ub` and `--threads`
- `--fit`
- speculative-decoding options other than `-md`

## Requests

### Endpoints

- **Chat:**
  - `POST /v1/chat/completions` (OpenAI)
  - `POST /v1/messages` (Anthropic)
  - `POST /v1/responses` (OpenAI Responses)
- **Text:** `POST /v1/completions` and llama.cpp's `POST /completion`.
- **Utilities:**
  - `POST /tokenize` and `/detokenize`
  - `POST /apply-template`, which renders a chat without running it
  - `POST /v1/chat/completions/input_tokens` and `/v1/messages/count_tokens`, which count a request's tokens
- **State:**
  - `GET /health`
  - `GET /v1/models`
  - `GET /props`
  - `GET /slots` and `GET /metrics`, which are answered even while a long prompt is being read
- **Web UI:** `GET /`, when the build embedded it.
- **Refused:**
  - embeddings and reranking
  - LoRA adapters
  - `POST /slots/0?action=save` and `restore`

  `action=erase` works and empties the slot.

### Sampling per request

A request can set:

- `temperature`
- `top_k` (1 to 256 at a temperature above 0)
- `top_p`
- `min_p`
- `seed`
- `max_tokens` (or `n_predict`)
- `stop`

Anything it leaves out comes from the launch settings.

- **Temperature 0** is greedy decoding, and it is deterministic.
- **At a temperature above 0** the same request with the same seed gives the same tokens.
- **Speculative decoding does not change what is sampled.** Each drafted token is accepted or rejected with the exact rule that keeps the output distributed as the target model's own (see [engine.md](engine.md#a-decode-step)).

The request is refused with `not supported by FNFL: ...`, naming each field, if it asks for any of these:

- repeat, frequency or presence penalties
- `ignore_eos`
- `logit_bias`
- `n_probs` or `logprobs`
- DRY, XTC, `typical_p`, `top_n_sigma`, mirostat, dynamic temperature or `adaptive_target`
- `min_keep`
- a custom sampler order
- a reasoning budget (`thinking_budget_tokens`)
- `n` greater than 1

**Constrained output is refused as well.** A request with `grammar`, `json_schema`, or a `response_format` of type `json_object` or `json_schema` gets the error `grammar and JSON-schema constrained output are not supported by FNFL`. The engine's sampler has no grammar stage.

Tool calls still work. The chat template's own tool-call grammar travels with the request but is not enforced; tool calls are parsed from the model's text by the template's parser, as in llama.cpp.

### One request at a time

There is one slot. A second request waits until the first ends.

The server keeps the conversation it last processed and reuses as much of it as the next prompt shares:

- **An exact extension** of what the slot holds, which is the normal case for a chat follow-up, processes only the new tokens.
- **A prompt that diverges inside the held conversation** restores the latest checkpoint before the divergence and re-reads from there.
  - The engine takes checkpoints at every multiple of 8,192 tokens and at the end of each prompt, keeping the newest eight.
  - When no checkpoint fits, the prompt is read from the start.
- **Chat clients re-send earlier replies as text, which the template may re-tokenize differently** (whitespace around `</think>`, tool-call formatting). The server recognizes a re-sent reply that matches one it generated and splices in its original tokens. That keeps follow-ups as extensions.
  - The log shows these as `BG1:` and `BG2:` lines.
  - This needs `cache_prompt`, which is on by default.

### Long prompts

The engine reads prompts in chunks of 8,192 tokens at about 3,500 tokens/s on a fresh prompt. A 130,000-token prompt therefore takes about 37 s before the first token.

- A streaming request gets its HTTP headers at once.
- With `"return_progress": true`, it also gets a progress event at the end of each chunk.
- If the client disconnects mid-prompt, the engine stops at the next chunk's end. It keeps what it has read, so a retry of the same prompt resumes there.

## Reading the log

The server writes to standard error. Besides llama.cpp's usual request lines, the engine adds three lines per request.

```
FNFL prefill: a pure extension at 33117: 2990 tokens processed (33117 kept by the shell, 2990 new; 1.124 s); chunks (start+rows, * checkpoint): 33117+2990*; checkpoints held 3
FNFL prefill phases (...): ... decode-ready ... (+1123.745); rows 2990; restored checkpoint none
FNFL request: ended eog; steps 8098 (bootstrap 1, speculative 8097, generate 0); tokens 26038; tokens a speculative step 3.216; accepted drafts a speculative step 2.216; step wall ms median 16.222 p90 16.435 ...; depth before a step 36107..62142; ... 198.12 tokens/s (all tokens), 198.11 tokens/s ((tokens - 1) / span: the server's predicted_per_second definition); closing check checked
```

- **`FNFL prefill`** says how the prompt was handled: a pure extension, a checkpoint restore or a fresh read. It gives the tokens processed, the time, the chunks (`*` marks a checkpoint) and the checkpoints held.
- **`FNFL request: ended`** gives:
  - why the reply ended:
    - `eog`: the model's end token
    - `budget`: `max_tokens` or the end of the context
    - `the shell (...)`: a stop string or a cancel
    - `the next request`: the reply was still open when the next request arrived
  - the tokens generated
  - **tokens a speculative step**, which drives speed: one plus the accepted drafts, at most `d + 1`
  - the median and 90th-percentile step time
  - the depth range
  - the decode rate
- **Failures** end the request with an error whose message starts with the engine's status: `FNFL_ERROR: ...` or `FNFL_BAD_REQUEST: ...`. After a failed step, the next request re-reads its prompt from the start.

## Memory

- **Each card** holds its half of the weights, 23.2 GiB, and about 16 KB of state per context token, 12 KB of which is the target's keys and values. It also holds fixed state, the prompt buffers and the CUDA graphs.
  - At 198,720 cells, 2.41 GiB is left free on each card.
  - At 262,144 cells, 1.47 GiB is left free.
  - The [engine description](engine.md#memory) lists the parts.
- **The host** memory-maps the two model files, 75.8 GB. After the load, only the per-layer-embedding table (26.9 GiB) is still read. Its pages stay in the page cache while memory allows, or permanently with `--fnfl-ple-table full`. The server process itself uses well under 1 GiB of anonymous memory.

## The development launch script

[`fn/tools/fnfl-serve.sh`](../../fn/tools/fnfl-serve.sh) starts, stops and reports on the server on the development machine. Its model, draft and subset paths and its GPU-lock directory are written into the script, so it is a template elsewhere.

- **`fnfl-serve.sh start [--temp T] [--top-k K] [--top-p P] [--min-p P] [--spec-d N] [--draft-attention dense|sparse]`**
  1. Refuses if any compute process is on either card or a lock is held.
  2. Takes the three GPU locks (`gpu.lock`, `gpu0.lock` and `gpu1.lock`) with `flock` and hands them to the server process, which holds them for its lifetime.
  3. Starts the server detached on all interfaces.
  4. Waits up to 600 s for `/health`.

  It records the binary's SHA-256 and the settings in `start.txt`. Its defaults are the serving settings: the full 262,144-token context, `--fnfl-spec-d 4` and `--fnfl-draft-attention sparse`. It always passes both FNFL settings to the server.
- **`fnfl-serve.sh stop`:** sends SIGTERM to the recorded PID only, waits for it to exit, then checks both cards are at 0 MiB and the locks are free.
- **`fnfl-serve.sh status`:** the PID, `/health` and the locks.
- **Environment:**
  - `BIN` (default `engine/build-fnfl/bin/llama-server`)
  - `SERVE_DIR` (where `server.log`, `server.pid` and `start.txt` go)
  - `PORT` (8190)
  - `CTX` (262144)

## Troubleshooting

- **The load fails with a shape, architecture or tensor message.** The file is not the GSQ-RCO IQ3_XXS Qwen3.8-Flash-Next, or `-m` names the second shard. Point `-m` at `...-00001-of-00002.gguf`.
- **The load fails with "cannot read the draft subset".** Pass `--fnfl-draft-subset` with the repository's file.
- **The load fails while reserving memory.** Another process holds memory on one of the cards, or `-c` is too large for it. Check `nvidia-smi`.
- **The load fails when it enables peer access, or the first request fails in a card-to-card exchange.** The two cards cannot reach each other directly. Check `nvidia-smi topo -p2p r`. On AMD boards, boot with `iommu=pt`.
- **The first replies are slower than later ones.** With the lazy table, the first requests read per-layer-embedding pages from disk. Either let the page cache warm up or use `--fnfl-ple-table full`.
- **A request is refused with `not supported by FNFL`.** Remove the named field. Many clients send `presence_penalty`, `frequency_penalty` or `logprobs` only when they are configured to.
- **A request is refused with `grammar and JSON-schema constrained output are not supported by FNFL`.** The client asked for structured output. Turn off its structured-output or JSON mode.
