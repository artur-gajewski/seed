"""MPS replication section for the combined, offline research report."""

import json
import statistics


def render_mps(root, chart, bars, esc):
    runs = []
    cpu = []
    for scale, seeds in [("", [42, 43, 44]), ("full-scale", [42])]:
        for seed in seeds:
            for kind in ["transformer", "gru"]:
                filename = f"run-{kind}-{seed}.json"
                row = json.loads((root / "mps" / scale / filename).read_text())
                row["scale"] = scale or "pilot"
                runs.append(row)
                old = json.loads((root / scale / filename).read_text())
                old["scale"] = scale or "pilot"
                cpu.append(old)
    chunks = [
        "<h2>MPS replication and CPU comparison</h2><p>All eight training jobs were repeated on MPS using exact copies of the CPU corpora, identical initial seeds, model sizes, 800-update budgets, and optimizer settings. CPU results remain preserved. No CPU fallback environment variable was enabled.</p>"
    ]
    summary = {}
    for scale, label in [
        ("pilot", "Small models: three paired seeds"),
        ("full-scale", "Original parameter scale: seed 42"),
    ]:
        chunks.append(
            f'<h3>{label}</h3><div class="scroll"><table><tr><th>Model / device</th><th>Train seconds</th><th>Task loss</th><th>Prose loss</th><th>Accuracy %</th><th>Decode tok/s</th></tr>'
        )
        chart_rows = []
        summary[scale] = {}
        for kind in ["transformer", "gru"]:
            for device, source in [("cpu", cpu), ("mps", runs)]:
                group = [r for r in source if r["scale"] == scale and r["kind"] == kind]

                def value(key, factor=1, group=group):
                    vals = [r[key] * factor for r in group]
                    avg = statistics.mean(vals)
                    return f"{avg:.3f}" + (
                        f" ± {statistics.stdev(vals):.3f}" if len(vals) > 1 else ""
                    )

                summary[scale][f"{kind}-{device}"] = {
                    key: statistics.mean(r[key] for r in group)
                    for key in [
                        "training_seconds",
                        "task_loss",
                        "prose_loss",
                        "accuracy",
                        "decode_tokens_s",
                    ]
                }
                chunks.append(
                    f"<tr><td>{kind} / {device.upper()}</td><td>{value('training_seconds')}</td><td>{value('task_loss')}</td><td>{value('prose_loss')}</td><td>{value('accuracy', 100)}</td><td>{value('decode_tokens_s')}</td></tr>"
                )
                chart_rows.append(
                    (
                        f"{kind} / {device.upper()}",
                        statistics.mean(r["training_seconds"] for r in group),
                    )
                )
        chunks.append(
            "</table></div><p>Mean ± sample standard deviation for three seeds; original-scale values are single observations. Training timing synchronizes MPS at update boundaries and excludes batching/data transfers, validation, saving and startup. Small GPU workloads can be dominated by dispatch and synchronization overhead.</p>"
        )
        chunks.append(bars(f"{label}: fixed-budget update time", chart_rows, "s"))
        for kind in ["transformer", "gru"]:
            values = summary[scale]
            ratio = (
                values[f"{kind}-cpu"]["training_seconds"]
                / values[f"{kind}-mps"]["training_seconds"]
            )
            chunks.append(
                f"<p>{kind.upper()} CPU/MPS training-time ratio: <strong>{ratio:.2f}×</strong>. Above one means MPS was faster; below one means CPU was faster under this protocol.</p>"
            )
        series = []
        for device, source in [("cpu", cpu), ("mps", runs)]:
            for kind in ["transformer", "gru"]:
                group = [r for r in source if r["scale"] == scale and r["kind"] == kind]
                points = [
                    (
                        statistics.mean(
                            r["history"][i]["train_seconds"] for r in group
                        ),
                        statistics.mean(r["history"][i]["mixed_loss"] for r in group),
                    )
                    for i in range(len(group[0]["history"]))
                ]
                series.append((f"{kind} {device}", points))
        chunks.append(
            chart(
                f"{label}: learning vs. update time",
                series,
                "Update seconds",
                "Mixed validation loss",
            )
        )
    original = summary["full-scale"]
    transformer_ratio = (
        original["transformer-cpu"]["training_seconds"]
        / original["transformer-mps"]["training_seconds"]
    )
    gru_ratio = (
        original["gru-cpu"]["training_seconds"]
        / original["gru-mps"]["training_seconds"]
    )
    chunks.insert(
        1,
        f'<div class="card"><strong>Device result</strong><p>At the original parameter scale, MPS completed the Transformer update budget {transformer_ratio:.2f}× faster and the GRU budget {gru_ratio:.2f}× faster than CPU. These single-seed scale results need replication. The small-model device comparison below shows that GPU acceleration is workload-dependent. The small models and original-scale GRU remain close to the 14.3% constant-color baseline. The original-scale MPS-trained Transformer is an exception, examined below. Faster training alone is not evidence of better reasoning or lower electrical energy.</p></div>',
    )
    chunks.insert(
        2,
        '<div class="card"><strong>Recommendation for Seed</strong><p>Use the Transformer on MPS as the default for training at roughly 3.3 million parameters. It trained faster and produced the strongest task result in this experiment.</p><ul><li><strong>Normal training:</strong> Transformer + MPS.</li><li><strong>Small, quick experiments:</strong> CPU—the GPU offered little benefit at that scale.</li><li><strong>GRU:</strong> Keep as a research baseline. Its lower prose loss did not translate into reliable color retrieval. CPU was faster for GRU generation.</li><li><strong>Quantization and pruning:</strong> Keep optional. Our int8 implementation reduced storage but slowed decoding; pruning damaged quality.</li></ul><p>Before treating the 88.5% accuracy as dependable, repeat the larger Transformer experiment with seeds 43 and 44 on both devices. That is the most useful next experiment.</p><p>These recommendations apply to the tested 128-byte context. Seed’s usual 256-byte context still needs benchmarking.</p></div>',
    )
    cross_device = json.loads((root / "mps/cross-device.json").read_text())
    chunks.append(
        '<h3>Unexpected Transformer accuracy improvement: cross-device check</h3><p>The original-scale MPS-trained Transformer achieved 88.5% exact-answer accuracy versus 13.2% for the CPU-trained seed-42 model, despite similar prompt-inclusive losses. Each saved checkpoint was then evaluated on both CPU and MPS. The accuracy follows the trained weights: moving either checkpoint to the other device preserves its score. This rules out an inference-device explanation for the observed score difference in these evaluations. Training numerics can lead to different optimization trajectories, but this experiment does not isolate the mechanism or establish that MPS generally learns better. This is one paired seed; additional initializations are needed.</p><div class="scroll"><table><tr><th>Trained on</th><th>Evaluated on</th><th>Task loss</th><th>Prose loss</th><th>Accuracy %</th></tr>'
    )
    for row in cross_device:
        chunks.append(
            f"<tr><td>{row['trained_on'].upper()}</td><td>{row['evaluated_on'].upper()}</td><td>{row['task_loss']:.6f}</td><td>{row['prose_loss']:.6f}</td><td>{100 * row['accuracy']:.1f}</td></tr>"
        )
    chunks.append(
        "</table></div><p>The MPS-trained large Transformer scored 78.6% on direct questions, 89.3% on distractor questions, and 89.3% on state updates. These use familiar templates with held-out color/object combinations, not a broad reasoning benchmark. Raw cross-device metrics and every completion are saved under <code>mps/cross-device.json</code> and <code>mps/cross-answers-*.json</code>. Reproduce with <code>.venv/bin/python research/cross_device.py</code> outside the sandbox.</p>"
    )
    chunks.append(
        '<h3>MPS-trained inference interventions</h3><p>Pruning is timed on MPS against its MPS parent. Int8 GRU is CPU-only: the table includes a fresh CPU float32 measurement of exactly the same MPS-trained weights, so its quantization comparison does not mix devices. Serialized state sizes exclude optimizer state. These interventions use seed 42 only.</p><div class="scroll"><table><tr><th>Scale / variant</th><th>Device</th><th>State KiB</th><th>Task loss</th><th>Prose loss</th><th>Accuracy %</th><th>Decode tok/s</th></tr>'
    )
    for parent in runs:
        if parent["seed"] != 42:
            continue
        candidates = [parent]
        if "cpu_parent" in parent:
            candidates.append(
                dict(parent["cpu_parent"], state_bytes=parent["state_bytes"])
            )
        candidates.extend(parent["variants"])
        for r in candidates:
            chunks.append(
                f"<tr><td>{esc(parent['scale'])} / {esc(r['kind'])}</td><td>{r['device'].upper()}</td><td>{r['state_bytes'] / 1024:.1f}</td><td>{r['task_loss']:.3f}</td><td>{r['prose_loss']:.3f}</td><td>{100 * r['accuracy']:.1f}</td><td>{r['decode_tokens_s']:.0f}</td></tr>"
            )
    chunks.append(
        "</table></div><p>Generation carries GRU state; the Transformer has no KV cache. Both include device execution but benchmark decoding leaves predicted token tensors on device, whereas quality evaluation copies token IDs to CPU to implement stopping. Numerical differences between CPU and MPS are expected; identical seeds do not promise identical training trajectories. A low prose or prompt-inclusive task loss still does not demonstrate reliable answer retrieval.</p><h3>MPS memory and limitations</h3><p>Peak RSS is process host memory, not peak GPU memory. The MPS runs additionally record live tensor allocations and driver allocations after benchmarking; those are snapshots, not peaks. Driver allocations can include cached buffers and framework allocations. No watts or joules were measured. CPU and GPU runs were conducted sequentially; host activity and thermal state were not fully controlled.</p>"
    )
    chunks.append(
        '<div class="scroll"><table><tr><th>Scale / model / seed</th><th>Host peak RSS MiB</th><th>MPS live MiB</th><th>MPS driver MiB</th></tr>'
    )
    for r in runs:
        m = r["mps_memory"]
        chunks.append(
            f"<tr><td>{r['scale']} / {r['kind']} / {r['seed']}</td><td>{r['peak_rss_bytes'] / 1048576:.1f}</td><td>{m['allocated_bytes_after_benchmark'] / 1048576:.1f}</td><td>{m['driver_allocated_bytes_after_benchmark'] / 1048576:.1f}</td></tr>"
        )
    chunks.append(
        "</table></div><p>Reproduce with <code>.venv/bin/python research/mps_run.py</code> outside the GPU-restricting sandbox, followed by <code>.venv/bin/python research/cross_device.py</code> and <code>.venv/bin/python research/report.py</code>. Raw MPS metrics, answers, corpora and checkpoints are under <code>research/mps/</code>. GPU progress and backend messages are in <code>mps-progress.jsonl</code> and <code>mps-stderr.txt</code>.</p>"
    )
    (root / "mps" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return "".join(chunks), {
        "runs": runs,
        "cross_device": cross_device,
        "environment": json.loads((root / "mps/environment.json").read_text()),
        "summary": summary,
    }
