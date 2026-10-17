# fnfl-drive, the developer driver

`fnfl-drive` runs pieces of the FNFL engine, or the whole model, without the server. It is the engine's test and benchmark harness. Each kernel family has a test against a CPU reference and a bench, and the whole-model commands load the model the same way the server does. Its full usage is the comment at the top of [`fn/tools/fnfl_drive.cpp`](../../fn/tools/fnfl_drive.cpp). This page groups the commands and explains what they need.

It is a developer tool built around the development machine, so read the next section before using it anywhere else.

## Before you run it

- **The model paths are built in.** There is no `--model` option.
  - The target, the draft and the default draft subset are set in [`fn/runtime/target_cmds.h`](../../fn/runtime/target_cmds.h) (the `model_path`, `draft_path` and `draft_subset` members of `fnfl_target_args`).
  - To use other locations, edit those three lines and rebuild, or create the same paths on your machine.
  - `--draft-subset FILE` overrides the subset on the commands that take it.
- **Devices:** most GPU commands take device numbers (default 0 and 1), and the whole-model commands use devices 0 and 1. A device given twice runs a two-card test as a loopback on one card. That mode is meant for running under Compute Sanitizer.
- **Exclusive cards:** the whole-model commands reserve nearly all of both cards' memory, so nothing else may be running on them, including the server.
  - The driver itself takes no locks.
  - On the development machine, GPU work holds three `flock` locks (`gpu.lock`, `gpu0.lock` and `gpu1.lock`), as `fn/tools/fnfl-serve.sh` does.
- **Token-ID files:** where a command takes an `IDS`, `PROMPT_IDS` or `HISTORY_IDS` file, it is a list of token IDs in any separator: commas, spaces, newlines or a JSON array. Make one with the server's `/tokenize` endpoint.
- **Output:** `OUT` is a directory the command writes its logs, tokens and tables into.
- **Exit status:** 0 when every check passes.
- **Options anywhere after the command:**
  - `--ple-table lazy|full` sets the PLE table's residency for any command that loads the model, as for the server.
  - `--draft-attention dense|sparse` turns on the draft and chooses its attention for the commands that run it.

## CPU-only checks

These need no GPU and no model.

- **`fnfl-drive sampler-ref`:** the CPU reference sampler's own tests.
- **`fnfl-drive grid-test [HEADER]`:** the engine's quantization grid tables against ggml's `ggml-common.h`. Pass `ggml/src/ggml-common.h`, since the default is a development-machine path.
- **`fnfl-drive synth-check`:** the synthetic weight-block generators that the kernel tests use.
- **`fnfl-drive decode-check DIR`:** the CPU weight decoders, bitwise against a saved dump of the pinned llama.cpp release's decoders.
- **`fnfl-drive ex-prep [CACHE]`, `ep-prep [REFS] [CACHE] [ROUTES]`:** precompute the synthetic expert weights and FP64 references that `ex-test` and `ep-test` read.

## Kernel tests and benches

Each family has a `-test` that checks the kernels against an FP64 CPU reference, or bitwise where the operation is exact, and a `-bench` that times them. Most run on synthetic data. A few read one or two layers' real tensors from the model file without loading the model; these are marked "model file".

- **The decode activation format:** `act-test`, `act-bench`.
- **Hyper-connections:**
  - `hc-test` and `hc-bench` on one card;
  - `hc2-test` and `hc2-bench` across both cards, with the exchange.
- **GDN:** `gd-test` and `gd-bench` (model file: layers 10 and 40).
- **QSA:**
  - `qa-test` and `qa-bench`, the indexer and block selection;
  - `qa2-test` and `qa2-bench`, the attention (model file: layer 11).
- **Routed and shared experts:**
  - `ex-calib`, `ex-test` and `ex-bench` at decode rows;
  - `ep-test` and `ep-bench` at prompt rows.
- **Dense quantized products:** `dp-test`, `dp-bench`.
- **The token embedding and PLE:**
  - `ple-test` and `ple-bench` (model file);
  - `ple-pin-test`, for the table's pinned residency on a temporary file.
- **The output head, logits and greedy pick:** `v-test`, `v-bench`.
- **The draft block's kernels:** `draft-kernel-test`, `draft-kernel-bench`.
- **The sampler on both cards against the CPU reference:** `samp-test`, `samp-bench`.
- **The card-to-card protocol:** `xchg-test`, `xchg2-test`.
- **The prompt-path kernels:** `pd-test`, `hcp-test`, `gdp-test`, `qsp-test`, `pe-test` and `dcp-test`. These cover the dense products and router, HC and its exchange, GDN, QSA, the embedding and PLE, and the draft's catch-up pieces.
- **Prototypes:** `px1-link`, `xp1-test` and `xp1-bench`, kept from the design's prototyping.

Most tests take `san` for a smaller case set suited to Compute Sanitizer. Most benches take `ncu` or `nsys` to run one point for a profiler.

## Whole-model commands

These open the real model files at the built-in paths.

### Without loading weights

- **`fnfl-drive load-plan [OUT] [MODEL] [DRAFT]`:** CPU only. It prints the placement of every tensor slice on the two cards, the slice hashes and a dry run of the load.
- **`fnfl-drive model-dry IDS OUT [graph|eager|sample|spec] [--draft-attention dense] [--temp T] [--top-k K] [--top-p P] [--min-p P]`:**
  - It reserves everything and runs the whole target pass with zero weights: the CUDA graphs (`graph`) or eager launches (`eager`), the sampler (`sample`), or speculative rounds (`spec`).
  - With `--draft-attention`, it also runs the draft rounds.
  - This is the quickest check that a build's plumbing works on both cards. It reads the model files for their layout but copies no weights to the cards.
- **`fnfl-drive state-test [all|san] [graph|eager] [DEV0] [DEV1]`:** commit-by-version and truncation through the round frame, with random weights and no model load.

### With the model loaded

Each of these loads the full model once.

- **`fnfl-drive load-check [OUT] [CELLS] [MODEL] [DRAFT]`:** loads the model at `CELLS` (default 198,720) and checks that every byte reached its place on both cards. It reports the memory account.
- **`fnfl-drive generate PROMPT_IDS OUT [N_TOKENS] [--temp T] [--top-k K] [--top-p P] [--min-p P] [--seed S]`:** prefills the prompt and generates without speculation, one token a step. It is greedy unless a sampling option is given.
  - `--evict-ple` evicts the PLE table from the page cache first.
  - `--dry-paths` checks the paths without loading.
- **`fnfl-drive spec PROMPT_IDS OUT run --draft-attention dense|sparse [--spec-d N] [--runs N] [--eog ID,ID..] SETTING [--and SETTING ..]`:** the runtime engine exactly as the server drives it.
  - `SETTING` is `[--temp T] [--top-k K] [--top-p P] [--min-p P] [--seed S] [--max-tokens N]`, and `--and` starts another setting.
  - Each setting runs `--runs` requests, each from a fresh prefill.
  - It writes `run-summary.txt` and `rounds.csv`, one row a step with its depth and timing.
  - `run-dry` does the same with zero weights.
  - **This is the right tool for a matched decode comparison**, since it gives the same prompt, settings and seed in one process.
- **`fnfl-drive spec HISTORY_IDS OUT ab|ab-dry --draft-attention sparse [--arms D:FORM[:EX],..] SETTING`:**
  - It prefills a history once, saves the decode-ready state, and runs one continuation per arm from that same state.
  - An arm is a proposal count, a draft attention, and optionally an expert-kernel mapping, for example `2:dense,4:sparse`.
- **`fnfl-drive prefill-test IDS OUT [plan|dry|dry-eager|vcheck|model] [POSITIONS] --draft-attention dense ...`:**
  - It checks the bulk prompt path against the decode path.
  - `plan` shows the chunking without a device. `dry` and `dry-eager` use zero weights. `model` runs the real model, with optional deep-restore and append stages.
- **`fnfl-drive model-bench IDS OUT [all|nsys [DEPTH B [real]]]`:**
  - It times a forced target pass at 4,096, 58,581, 131,072 and 196,608 cells, or one point for Nsight Systems.
  - This is a diagnostic of the target pass alone, not a measure of generation speed.
- **`fnfl-drive model-test`, `rollback-test`, `layer-test`, `layer-bench`, `draft-test` and `draft-bench`:** these compare the engine against reference dumps produced by the pinned llama.cpp release with the project's dump tools. Those dumps are not part of this repository.

## Typical uses

```sh
# does a fresh build run on both cards? (zero weights, no model data)
build-fnfl/bin/fnfl-drive model-dry prompt.ids out/dry graph --draft-attention dense

# one kernel family after a change, and again under Compute Sanitizer
build-fnfl/bin/fnfl-drive dp-test 0 all full
compute-sanitizer build-fnfl/bin/fnfl-drive dp-test 0 all san

# two decode settings on the same prompt, three requests each, as the server would run them
build-fnfl/bin/fnfl-drive spec prompt.ids out/spec run --draft-attention sparse --spec-d 4 --runs 3 \
--temp 1 --top-k 20 --top-p 0.95 --min-p 0.05 --seed 1 --max-tokens 512 \
--and --temp 0 --max-tokens 512
```
