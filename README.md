# llama.cu

A minimal, from-scratch CUDA implementation of **LLaMA inference** — no
PyTorch, no Hugging Face, no Python runtime. A single `.cu` file that
loads a checkpoint, allocates a KV cache, and autoregressively decodes a
token on the GPU.

Built to demonstrate how a modern decoder-only transformer maps onto
GPU hardware: memory bandwidth, kernel launches, thread blocks, shared
memory, and cuBLAS — and what actually bottlenecks LLM decode.

![C++](https://img.shields.io/badge/language-C%2B%2B-blue)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-green)
![License MIT](https://img.shields.io/badge/license-MIT-lightgrey)

---

## What this proves

- **You can run a transformer without a framework.** Every matmul,
  RMSNorm, RoPE, attention head, and SwiGLU gate is explicitly launched
  as a kernel you can read and profile.
- **Decode is memory-bound, not compute-bound.** This code is structured
  around that insight: weights are streamed once (via cuBLAS GEMV),
  activations stay on device, and only the final logits cross PCIe.
- **Production-style CUDA hygiene:** RAII handles, checked
  `cudaMalloc` / `cudaMemcpy` / kernel launches, `mmap`-based
  checkpoint loading, and a single contiguous `device_weights_` buffer
  carved into named views — zero per-tensor allocations at runtime.

## Quick start

Requires a Linux or WSL host with an NVIDIA GPU and the CUDA Toolkit
≥ 11.0. Tested compute capabilities: **8.0 (Ampere), 8.6, 9.0 (Hopper)**.

```sh
# 1. Build (Makefile or CMake — both work)
make
# or
mkdir build && cd build && cmake .. && make -j

# 2. Run with the bundled TinyStories model (15M params)
./llama stories15M.bin tokenizer.bin 200 "I have a dream"

# args: [checkpoint] [tokenizer] [max_new_tokens] [prompt]
```

Expected output:

```text
I have a dream that my four little children will one day live in a nation
where they will not be judged...
Tokens: 47, time: 0.12s, speed: 392 tok/s
```

> The sample checkpoints (`stories15M.bin`, `tokenizer.bin`) are the
> small educational models exported by [llama2.c](https://github.com/karpathy/llama2.c).
> Any checkpoint in that binary format — `LlamaConfig` header followed by
> raw FP32 weights — loads without conversion. See
> [Checkpoint format](#checkpoint-format).

## Architecture

```
              ┌─────────────────────────────────────────────┐
  prompt ──►  │  BPE Tokenizer (CPU, vocab lookup + merge)   │
              └───────────────────┬─────────────────────────┘
                                  │ token ids
                                  ▼
              ┌─────────────────────────────────────────────┐
              │  Embedding lookup  (device scatter-copy)     │
              └───────────────────┬─────────────────────────┘
                                  │ x ∈ R^dim
              ┌───────────────────▼─────────────────────────┐
              │  for each layer l in 0..n_layers-1:           │
              │    RMSNorm(x) ──► Q, K, V  (cuBLAS GEMV)     │
              │                   RoPE(Q, K)                 │
              │                   GQA attention + KV cache   │
              │                   O-projection ──► x +=      │
              │    RMSNorm(x) ──► gate, up  (cuBLAS GEMV)    │
              │                   SwiGLU ──► down ──► x +=   │
              └───────────────────┬─────────────────────────┘
                                  ▼
              RMSNorm ──► LM head (cuBLAS GEMV) ──► logits
                                  │
                                  ▼  (only sync point per token)
                           argmax → next token → repeat
```

Key properties:

| Property | Choice | Why |
|---|---|---|
| Precision | FP32 (FP32 activations, FP32 weights) | Numerical parity with `llama2.c`; easy to extend to FP16/BF16 |
| Matmul | `cublasSgemv` (GEMV, not GEMM) | Decode is one vector × matrix per projection; GEMM would waste batch dims |
| Attention | GQA-aware kernel, one block per query head | Matches `n_kv_heads ≤ n_heads` (LLaMA 2/3 grouped-query attention) |
| KV cache | Pre-allocated `[n_layers × max_seq_len × kv_dim]` | No realloc during decode; cache adds O(seq) memory, O(1) per step |
| Position | RoPE (rotary) applied in-place to Q/K | No learned position table needed at inference |
| Sampling | Greedy `argmax` over logits on CPU | Deterministic baseline; swap in top-k / top-p / temperature |
| Host↔device traffic | 1× H2D (weights) + 1× D2H (logits) per token | Everything else stays resident on GPU |

## CUDA kernels

Each kernel is deliberately small so it can be reasoned about in isolation.
All launch wrappers check `cudaGetLastError()` immediately.

| Kernel | Launch config | What it does |
|---|---|---|
| `RmsNormKernel` | 1 block × 1024 threads | Per-row `x / sqrt(mean(x²)+ε) * w`. Uses `cub::BlockReduce` for the sum-of-squares across the whole row. Grid-stride loop handles any `dim`. |
| `ApplyRotaryEmbKernel` | `ceil(dim/2 / 128)` blocks | In-place RoPE rotation `[[c,-s],[s,c]]` on Q and K. Threads cover `dim/2` pairs; `kv_dim` guard handles GQA asymmetry. |
| `MultiHeadAttnKernel` | `n_heads` blocks × 1024 threads | Fused QKᵀ → scale → softmax → AV for one head. Softmax is numerically stabilized (max-subtract) and implemented with `cub::BlockReduce` + shared-memory broadcast. |
| `SwiGLUKernel` | `ceil(hidden_dim/128)` blocks | Elementwise `silu(gate) * up` where `silu(x)=x·sigmoid(x)`. Embarrassingly parallel; one thread per element. |
| `AccumKernel` | `ceil(dim/128)` blocks | Residual add `x += sublayer_out`. Separate kernel keeps the residual stream explicit. |
| `Matmul` (host wrapper) | cuBLAS | `y = Wᵀ·x` via `cublasSgemv(OP_T)`. Weight layout matches the checkpoint (row-major transposed), so no repacking is needed. |

Design notes worth knowing:

- **Why 1024 threads for RMSNorm/attention?** `dim` for LLaMA-scale
  models is 768–4096. A single 1024-thread block keeps the reduction
  inside one SM (no cross-block atomics) while the grid-stride loop
  covers any row size. For very large `dim` (>16k) this becomes the
  obvious multi-block target.
- **Why CUB?** `cub::BlockReduce` is a compile-time-tuned warp/block
  reduction — faster and simpler than hand-rolled tree reductions with
  `__syncthreads()` hazards, and it works for both sum and max in the
  softmax.
- **Where is the bottleneck?** During decode, each step streams
  *all* weights from HBM once (GEMV) and does O(seq·head_dim) work.
  Arithmetic intensity is ~1 FLOP/byte, so HBM bandwidth — not FLOPs —
  sets the ceiling. This is why quantization (INT8/FP8) and fused
  kernels are the highest-leverage next steps (see Roadmap).

## Source layout

Everything lives in a single translation unit, organised top-to-bottom
by authoring order rather than by class:

```
llama.cu
├── Error macros            CUDA_CHECK / CUBLAS_CHECK (fail-fast)
├── Config & state structs  LlamaConfig, LlamaLayerWeights, LlamaState
├── RAII helpers            CuBlasHandle (acquires/destroys handle)
├── LlamaModel              LoadWeights / AllocState / LayerForward / Forward
│   ├── Checkpoint loading  fopen header → mmap → cudaMalloc + 1× H2D copy
│   ├── Weight carving      contiguous buffer → per-layer named pointers
│   └── Inference loop      embed → N×LayerForward → norm → LM head
├── Device kernels          RmsNorm, RoPE, Attention, SwiGLU, Accum (+ wrappers)
├── Tokenizer               BPE encode/decode (CPU, llama2.c compatible)
├── Sampling helpers        ArgMax, SafePrint (UTF-8 aware), TimeInMs
└── Generate + main()       Prompt encode → prefill/decode loop → timed output
```

There is no build system ceremony: `make` invokes `nvcc` directly;
CMake is provided for IDE integration and multi-arch builds
(`CMAKE_CUDA_ARCHITECTURES 80 86 90`).

## Checkpoint format

The binary format matches `llama2.c` exactly (no conversion step):

```text
offset  size      field
------  -------   -----------------------------------------------------
0       7×int32   LlamaConfig: dim, hidden_dim, n_layers, n_heads,
                  n_kv_heads, vocab_size, max_seq_len
7×int32 ...       FP32 weights, in this fixed order:
                    embed_tokens, then per-layer grouped:
                      {attn_norm, q/k/v/o_proj, ffn_norm,
                       gate_proj, down_proj, up_proj} × n_layers,
                    final norm, freq_cis_real, freq_cis_imag (skipped
                    at load — RoPE is computed on-device),
                    lm_head (shared with embed if vocab_size < 0)
```

A negative `vocab_size` in the header means tied embeddings
(`lm_head == embed_tokens`).

The tokenizer file is the `tokenizer.bin` from `llama2.c`
(`max_token_length`, then `score + len + bytes` per vocab entry).

## Building details

```sh
# Makefile (fastest iteration)
make                    # nvcc -g -DUSE_CUBLAS=1 llama.cu -lm -lcublas
make clean

# CMake (multi-arch, IDE-friendly)
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

Environment variables you may want:

```sh
CUDA_VISIBLE_DEVICES=0 ./llama stories15M.bin   # pin to one GPU
export CUDA_LAUNCH_BLOCKING=1                  # serialize launches (debug)
```

## Extending this codebase

The cleanest next steps, in order of impact for decode throughput:

1. **FP16 / BF16 weights + Tensor Cores** — halve HBM traffic, use
   `cublasGemmEx` or WMMA for projections (prefill) and keep GEMV for
   single-token decode.
2. **INT8 / FP8 weight-only quantization** — dequantize on the fly in
   a fused GEMV kernel; the single biggest win for memory-bound decode.
3. **CUDA Graphs** — capture the per-token launch sequence to erase
   ~10–30 µs/token of CPU launch overhead at small `dim`.
4. **Paged / chunked KV cache + FlashAttention-style tiling** — for
   long contexts where the cache no longer fits comfortably in HBM.
5. **Sampling** — replace `ArgMax` with temperature + top-k/top-p
   (softmax on device, `cub::BlockReduce` for selection).
6. **Continuous batching** — serve multiple sequences with a paged
   attention scheduler instead of one greedy loop.

For profiling: `nsys` (kernel timeline / gaps between launches),
`ncu` (per-kernel HBM bandwidth, occupancy, warp stalls), and
Nsight Systems' `cudaLaunch` view to confirm the decode loop is
launch-bound at small models.

## Limitations (by design)

- FP32 only — no mixed precision yet.
- Greedy decoding only (`argmax`); no temperature / top-p.
- Single sequence, single GPU, no tensor/pipeline parallelism.
- No FlashAttention; attention is the straightforward
  `softmax(QKᵀ/√d)·V` with KV cache.
- Hard-coded `1e-5` RMSNorm epsilon and `10000.0` RoPE base (the
  LLaMA defaults).

These are intentional: the goal is a readable, correct baseline that
makes each of the above a well-scoped extension.

## Acknowledgements

Inspired by [llama2.c](https://github.com/karpathy/llama2.c) (checkpoint
format, tokenizer, reference CPU implementation) and the broader LLaMA
community. Sample weights are Karpathy's TinyStories educational models.

## License

MIT — see [LICENSE](LICENSE).
