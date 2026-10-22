# Seed efficiency research: pre-run plan

This is a bounded CPU pilot, not a claim about human learning or a full architecture search.

1. Generate byte-level color retrieval, distractor, and state-update lessons. Partition color/object pairs before generating examples (60% train, 20% validation, 20% test), keeping every answer color represented. Deduplicate existing short stories and assign complete stories by SHA-256 to 90/5/5 splits. Record source hashes and generated corpora.
2. Train the existing Seed Transformer (width 80, two layers, four heads) and a one-layer GRU (embedding 80, hidden 180) at roughly comparable parameter counts, with three paired seeds. Both use 128-byte contexts, batch eight, float32, AdamW, gradient clipping, 800 updates, identical sampled batches, alternating task/prose batches, and the same warmup/cosine schedule. Compare real counts rather than assuming equality. This is a reduced-scale pilot relative to the proposed 3.36M-parameter model.
3. Select no model using test data. Evaluate fixed validation windows every 100 steps. At completion evaluate held-out prose/task loss and greedy exact-answer task accuracy with per-family results. Report first validation loss <= 1.5 nats/byte as a prespecified capability threshold; no crossing means unreached.
4. On seed 42 only, physically remove half the Transformer's FFN channels by weight magnitude, without fine-tuning. This measures structured static pruning, not learned conditional sparsity. Dynamically quantize the GRU and output linear layer to int8 on CPU. Compare quality, serialized inference state size, and repeated inference timings to each parent.
5. Measure training update time (excluding evaluation and saving), end-to-end wall time, isolated-process peak RSS (includes interpreter and evaluation), forward throughput, and greedy generation throughput. Recurrent generation carries hidden state; Transformer recomputes its context without a KV cache. Report this implementation distinction. Timing is not an energy measurement.
6. Produce raw JSON, checkpoints, per-example answers, reproducible scripts, and one self-contained HTML report with inline SVG charts. No external assets are required.

MPS is probed and runs may be repeated with `--device mps` if available. CPU int8 experiments stay CPU-only. The local environment initially reports MPS unavailable. Full-scale replication, an SSM baseline, tuned equal-wall-time studies, energy instrumentation, and accelerator comparison are follow-up work, not completed claims.

Sources: https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_dynamic.html and https://docs.pytorch.org/docs/stable/notes/mps.html. Dynamic quantization supports GRU/Linear; Seed's raw ParameterDict Transformer needs a separate quantization implementation.

## Scale check added after the pilot

The completed pilot motivates one additional paired seed (42), keeping the same data, 800-update budget and optimizer. Use Transformer width 256, four layers, eight heads, and GRU embedding 256, hidden 896, near the original 3.36M parameter scale. Keep context 128 to isolate model-size changes; the original 256-byte-context workload remains untested. Store these runs in `full-scale/`. This addition is exploratory and has no multi-seed uncertainty estimate. No learning rate tuning or test-based checkpoint selection is performed. Repeat seed-42 inference interventions at this scale as well.

## MPS replication requested after diagnosing sandbox restrictions

The same PyTorch installation exposes MPS outside the sandbox, and a tensor operation succeeded there. Preserve CPU artifacts and repeat all eight training runs on MPS under `mps/`, using exact copies of the CPU corpora, seeds, budgets and model configurations. Synchronize GPU operations around timings. Run Transformer pruning on MPS; int8 GRU remains CPU-only and is compared with a newly timed CPU float32 copy of the same MPS-trained parent. Record post-benchmark MPS allocation snapshots without calling them peak memory. Generate a combined CPU/MPS HTML report. These are device replications, not an architecture or hyperparameter change.

## Post-run validation of the large Transformer accuracy gap

The MPS-trained large Transformer scored 88.5% versus the CPU-trained model's 13.2% at seed 42. This unexpected finding triggered a post hoc cross-device evaluation: reload each saved checkpoint independently and evaluate it on CPU and MPS, recording every answer. This is an inference validation of the existing checkpoints, not additional training or checkpoint selection. The gap follows trained weights on both devices. Multiple training seeds at the original scale remain necessary before attributing a systematic quality advantage to a backend.
