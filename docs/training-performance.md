# TypeScript reference training performance

This document records optimizations and historical CPU measurements of the retained
TypeScript reference engine. The current studio, API, and New life use PyTorch; see
[PyTorch setup and training](pytorch.md). These timings do not measure the current
GPU backend or establish a CPU/GPU speedup.

## Strategy and implementation

The reference engine retains its TypeScript Transformer, manual derivatives, Float64 weights,
AdamW optimizer, tokenizer, and version 2 checkpoint format. An optional Node-API addon calls Apple's
Accelerate BLAS directly on existing typed-array storage. This is **CPU acceleration**, not Metal
GPU training. The Linux reference engine can use OpenBLAS; otherwise it falls back to JavaScript.

Changes, in measured order:

1. Dense projections: use contiguous rows in JavaScript, or native `dgemm` for forward products,
   input derivatives, and accumulated weight derivatives. Native calls do not copy tensor buffers.
2. Attention: pack each head's Q/K/V once and use BLAS for its forward and backward matrix
   products. Preserve causal masking, scaled scores, and softmax derivatives. This deliberately
   uses some extra temporary storage to eliminate expensive scalar loops.
3. GELU: calculate the cube by multiplication and cache the forward tanh for backpropagation.
   Cache allocation happens only when a backward tape exists.

Batch windows still accumulate sequentially. Clipping, weight decay, optimizer moments, training
schedule, validation partition, and all architecture settings are unchanged. Floating-point
reduction order can differ with BLAS; cross-backend results are numerically equivalent, not
guaranteed bit-for-bit identical. Loading an existing checkpoint preserves weights and moments.

## Enable on Apple Silicon

Use Node 24 and macOS with Xcode Command Line Tools installed. Build the optional addon once:

```sh
npm run native:build
SEED_TENSOR_BACKEND=accelerate npm run benchmark
```

The build uses `xcrun clang++` and the system Accelerate framework; no new npm dependencies or
automatic downloads are needed. Node headers must be installed. The script searches the matching
node-gyp cache and common Node installation locations. Set `NODE_INCLUDE_DIR` explicitly if needed:

```sh
NODE_INCLUDE_DIR=/path/to/node/include/node npm run native:build
```

The reference benchmark reports its active numerical backend; building the addon does
not change the PyTorch API backend.
`SEED_TENSOR_BACKEND=auto` (default) uses the addon when present on macOS. Set it to `javascript`
to force the portable engine, or `accelerate` to require native acceleration and fail clearly if
unavailable. A present but unloadable addon reports its error rather than silently falling back.
Rebuild the ignored `native/accelerate.node` when changing machine architecture.

Run reference benchmarks directly on macOS to use Accelerate. The current Linux
Docker studio uses CPU PyTorch and cannot access the Mac GPU or macOS addon.

## Reproduce measurements

```sh
SEED_TENSOR_BACKEND=javascript npm run benchmark
SEED_TENSOR_BACKEND=accelerate npm run benchmark
SEED_TENSOR_BACKEND=accelerate npm run profile:training
```

The benchmark uses seed 42, deterministic byte windows, and constant learning rate 0.0003. Defaults
are width 256, four layers, eight heads, batch eight, and context 256: **3,357,442 parameters**.
These are benchmark defaults, not the current studio defaults of width 512, twelve
layers, sixteen heads, batch eight, and context 2048 (39,142,658 parameters).
It performs one warmup and three measured updates, prints individual times, and reports the median
and a 30,000-update projection as JSON. Run engines sequentially on an otherwise idle machine.
Configure `BENCH_SAMPLES`, `BENCH_WARMUP`, `BENCH_WIDTH`, `BENCH_LAYERS`, `BENCH_HEADS`,
`BENCH_BATCH`, and `BENCH_CONTEXT` as environment variables. For example:

```sh
BENCH_WARMUP=2 BENCH_SAMPLES=7 SEED_TENSOR_BACKEND=accelerate npm run benchmark
```

The operation profiler runs the same full-size workload on an instrumented temporary copy of the
engine. It reports forward/backward kernel and optimizer times without adding production overhead.
It uses wall-clock timers: Node's sampling profiler terminated with signal 4 when profiling native
Accelerate on this host, so it is not used for native operation measurements.

Timing excludes model initialization, corpus encoding, periodic validation, checkpoint writes,
HTTP scheduling, and competing work. These are projected compute times, not completed 30,000-step
training runs; temperature, thermal conditions, thread settings, and corpus-dependent values can
affect results. The benchmark never loads or overwrites application checkpoints.

## Historical measurements

The following timings were recorded on October 16, 2026, on an Apple M4 Max,
arm64 macOS, Node 24.18.0, with the configuration above. They document earlier
optimization work and do not measure the current revision. The raw benchmark
results are no longer included in the repository. Run the benchmark commands
above to measure the current engine on your machine.

| Implementation                                    | Median ms/update | Projected hours / 30,000 updates |
| ------------------------------------------------- | ---------------: | -------------------------------: |
| Original JavaScript                               |           40,306 |                           335.89 |
| Contiguous projection loops                       |           11,674 |                            97.28 |
| Accelerate projections                            |            1,859 |                            15.49 |
| Final JavaScript fallback, including cached GELU  |           11,102 |                            92.52 |
| Accelerate projections and attention, cached GELU |              383 |                             3.19 |

The final native row uses two warmups and seven measured updates, run after the other benchmarks
finished. Earlier rows use one warmup and three measured updates. These are local measurements,
not universal throughput guarantees. The final result is about 105× faster than the local original
baseline. GPU timing is unavailable.

The original sampling profile attributed 82.18% to projection backpropagation, 13.29% to
projection forward passes, 3.06% to attention, and 1.25% to GELU. Allocation/GC and AdamW were
minor costs, so an allocator rewrite or optimizer changes were not justified.

Wall-clock operation timings (one warmup, one instrumented update; milliseconds):

| Operation                              | Accelerate projections only | Final implementation |
| -------------------------------------- | --------------------------: | -------------------: |
| Projection forward / backward          |                 36.8 / 68.5 |          34.3 / 66.5 |
| Attention forward / backward           |               523.5 / 695.1 |          63.8 / 40.9 |
| GELU forward / backward                |               241.0 / 260.0 |         105.5 / 16.2 |
| Layer normalization forward / backward |                 12.9 / 11.2 |          12.2 / 10.8 |
| Cross entropy                          |                         6.7 |                  6.3 |
| AdamW, including clipping              |                        22.5 |                 22.4 |
| Entire update                          |                     1,889.7 |                396.4 |

## Correctness checks

`npm test` and `SEED_TENSOR_BACKEND=javascript npm test` cover finite-difference gradients,
causal masking, learning from scratch, batch accumulation, legacy checkpoint loading, and exact
same-backend checkpoint resumption. Additional tests compare rectangular projection outputs and
nonzero accumulated derivatives against the original scalar kernel, validate native buffer bounds
and aliasing, and compare full model outputs, gradients, AdamW weights/moments, and resumed updates
between JavaScript and Accelerate with tolerance `1e-10 * (1 + abs(reference))`.

A separate historical comparison against the original engine snapshot checked 63,581 numeric values across
logits, gradients, three updates, and checkpoint state. JavaScript matched exactly on that fixture;
Accelerate's largest absolute difference was `7.77e-16`. This is a short-run numerical check, not
a guarantee of identical long-run trajectories. Native tests skip when the optional addon is absent;
build it before running the native verification suite.

## Historical acceleration alternatives

- **Apple Accelerate (implemented):** system-provided optimized CPU linear algebra, direct Float64
  buffers, a small native boundary, and no checkpoint conversion. Apple's documentation identifies
  it as CPU acceleration and documents double-precision GEMM.
  [Accelerate](https://developer.apple.com/documentation/accelerate),
  [dgemm](https://developer.apple.com/documentation/accelerate/cblas_dgemm).
- **MLX / Metal:** a promising next step if further speed is required. MLX offers GPU execution,
  automatic differentiation, and graph optimization, but adopting it would require mapping Seed's
  complete training graph, optimizer, and checkpoint state. Precision must be validated explicitly:
  MLX's NumPy conversion documentation says float64 arrays default to float32. Community Node
  bindings exist, with their own platform and build requirements; they are not a drop-in Seed engine.
  [Apple MLX](https://github.com/ml-explore/mlx),
  [dtype conversion](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/numpy.rst),
  [MLX-Node](https://github.com/mlx-node/mlx-node).
- **OpenBLAS (implemented for the reference engine):** optional native CPU matrix
  multiplication on Linux. See the native build scripts and tests; Raspberry Pi
  performance has not been measured here.
- **Workers:** could improve UI/API responsiveness or distribute batch windows, but gradient
  reduction and extra model copies add complexity. They do not replace efficient matrix kernels.

These historical optimizations did not include GPU training. The current PyTorch
backend supports MPS/CUDA, but no matched GPU comparison is recorded here and no
GPU speedup is claimed. The measured CPU improvement is sufficient to justify the smaller initial change.
