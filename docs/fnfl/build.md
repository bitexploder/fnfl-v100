# Building FNFL

FNFL builds with CMake and Ninja from one preset, `fnfl`. That preset builds llama.cpp's server, tokenizer and chat code on the CPU side and FNFL's own CUDA engine (`fn/`) for Volta (`sm_70`) only. No ggml GPU backend and no ggml CPU backend are built. The server never runs a ggml graph.

## Toolchain

The preset was written for, and is tested on, this toolchain:

- **CUDA 12.8** (12.8.93) with NVIDIA driver 580.
  - Any CUDA 12.x that supports GCC 14 should work.
  - CUDA 13 cannot build it, since it dropped `sm_70`.
  - Driver 580 is the last series that supports Volta.
- **GCC and G++ 14** (14.3.0). nvcc uses `g++-14` as its host compiler.
- **CMake 3.23 or newer** (4.2.3 tested), for preset version 4.
- **Ninja** (1.13.2 tested).
- **Python 3**, for the include check that runs with every build.
- **The OpenSSL development headers**, which are optional. Without them the server builds without HTTPS support, with a CMake warning.
- **Linux on x86-64** (Ubuntu 26.04.1, kernel 7.0 tested).

## Configure and build

```sh
cmake --preset fnfl
ninja -C build-fnfl llama-server fnfl-drive
```

The preset puts the build in `build-fnfl/`. To build somewhere else, add `-B DIR` to the `cmake` line and point Ninja at that directory.

### On a machine other than the development one

The preset names absolute paths from the development machine. Override any that differ on your system with `-D` on the configure line; command-line values take precedence over the preset's.

```sh
cmake --preset fnfl \
-DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14 \
-DLLAMA_UI_ARCHIVE= -DLLAMA_UI_ARCHIVE_SHA256=
```

These are the settings that matter:

- **`CMAKE_CUDA_COMPILER`** is `/usr/local/cuda-12.8/bin/nvcc` in the preset.
- **`CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER` and `CMAKE_CUDA_HOST_COMPILER`** are `gcc-14`, `g++-14` and `/usr/bin/g++-14` in the preset.
- **`LLAMA_UI_ARCHIVE` and `LLAMA_UI_ARCHIVE_SHA256`** pin the web UI to an archive on the development machine, with its SHA-256.
  - A configured archive that is missing or has the wrong hash stops the build.
  - Set both to empty to fall back to llama.cpp's own provisioning, in this order:
    1. a prebuilt UI already extracted in `tools/ui/dist/`;
    2. a download of the prebuilt UI from llama.cpp's Hugging Face bucket;
    3. if neither works, no embedded UI. CMake prints a warning and the server still builds.
  - To embed a UI offline, extract the `dist` of a llama.cpp release's UI into `tools/ui/dist/`.
  - The UI is a convenience. The API does not depend on it.

### What the preset sets

- `CMAKE_BUILD_TYPE=Release` and `CMAKE_CUDA_ARCHITECTURES=70`.
- `BUILD_SHARED_LIBS=ON`, with an `$ORIGIN` run path, so the binaries find their libraries beside them in `bin/`.
- `GGML_CPU=OFF`, `GGML_BACKEND_DL=OFF` and `GGML_OPENMP=OFF`. ggml is built only as the base library that the tokenizer and GGUF reader need.
- `LLAMA_BUILD_SERVER=ON`, `LLAMA_BUILD_UI=OFF` (no npm build) and `LLAMA_USE_PREBUILT_UI=ON`.
- `LLAMA_OPENSSL=ON`.
- `FETCHCONTENT_FULLY_DISCONNECTED=ON`. CMake's FetchContent makes no network fetches. This does not cover the web UI download above, which CMake's own download command makes.

### Other options

- **`-DFNFL_WERROR=ON`** turns the engine's warnings (`-Wall -Wextra`, for both host code and nvcc) into errors. The project's own checks build with it on.
- **The `fnfl-dev` preset** is `fnfl` with ccache on (`GGML_CCACHE=ON`), for repeated rebuilds. It builds in `build-fnfl-dev/`.

## Targets

- **`llama-server`:** the server. It needs the shared libraries beside it in `build-fnfl/bin/`: `libllama-server-impl.so`, `libllama-common.so`, `libllama.so`, `libggml.so` and `libggml-base.so`. To install it elsewhere, copy the whole `bin/` directory.
- **`fnfl-drive`:** the engine's developer driver. It runs kernel tests and benches and whole-model checks without the server ([fnfl-drive.md](fnfl-drive.md)).
- **`fnfl-chat-replay`:** replays raw generated text through the server's chat parsing and streaming code, CPU only. It is a test of the tool-call and reasoning parser against recorded cases.
- **`fnfl-splice-test`:** a self-contained test of how the server splices earlier generated replies back into a chat prompt. It needs no model.
- **`fnfl-check-includes`:** runs `fn/tools/check-includes.py` over the tree. It runs automatically before every engine library builds and fails the build on a violation.
  - The server may reach the engine only through `fn/fnfl.h`.
  - Engine code may reach borrowed llama.cpp and ggml CPU code only through `fn/borrowed/`.
  - No engine target may link a ggml GPU kernel.

## Runtime libraries

`llama-server` links the CUDA runtime dynamically (`libcudart.so.12`). The machine that runs it needs that library on the loader's path, from the CUDA toolkit or from the CUDA runtime package, and an NVIDIA driver that supports Volta.

## Checking a build

These need no model and no GPU:

```sh
build-fnfl/bin/fnfl-splice-test
build-fnfl/bin/fnfl-drive sampler-ref
build-fnfl/bin/fnfl-drive grid-test ggml/src/ggml-common.h
build-fnfl/bin/fnfl-drive synth-check
```

Each exits 0 when its checks pass. `grid-test` compares the engine's quantization grid tables with ggml's. Give it this tree's header, which is identical to the pinned one; without an argument it looks for a path on the development machine.

With the two cards, the driver's kernel tests, such as `act-test`, `hc-test` and `dp-test`, check single kernels against CPU references on synthetic data.

`fnfl-drive model-dry` runs the whole model's plumbing with zero weights, including the CUDA graphs, the card-to-card exchanges and the speculative rounds. It reads the model files for their layout but copies no weights to the cards. With real weights, start the server and send one request ([server.md](server.md)).

[fnfl-drive.md](fnfl-drive.md) covers the full set of checks and the driver's hard-coded model paths.
