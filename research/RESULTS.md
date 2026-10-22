# Seed efficiency research results

The [research folder](README.md) contains reproducible Transformer/GRU experiments,
training and evaluation corpora, raw results, and a [standalone HTML report](report.html)
with charts. Download the HTML and open it locally; it needs no server or network connection.
The study includes sixteen CPU/MPS training runs and eight pruning/quantization variants.
The GRU is a research baseline, not a selectable architecture in the studio.

- At roughly 3.3 million parameters, MPS completed the Transformer training budget **2.23×
  faster** and the GRU budget **1.96× faster** than CPU. Small models gained little or no benefit.
- The larger MPS-trained Transformer reached **88.5%** held-out color-retrieval accuracy,
  versus **13.2%** for the CPU-trained run. Evaluating both checkpoints on both devices
  confirmed that the difference follows the trained weights, not the inference device.
- GRU achieved lower prose loss but weak color retrieval. Int8 reduced storage but slowed
  CPU decoding; pruning reduced size while worsening quality.

Use Transformer + MPS as a provisional default for training at this scale, and CPU for
small experiments. The larger-model results use one paired seed and a **128-byte context**;
repeat across seeds and benchmark the usual 256-byte context before generalizing.
Electrical energy was not measured. See the [research plan](PLAN.md) and
[validation record](VALIDATION.md) for the protocol and checks.
