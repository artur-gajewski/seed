"""Render measured results into one offline HTML file with inline SVG figures."""

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COLORS = ["#3867e8", "#ce6419", "#21856b", "#a146a0"]


def esc(value):
    return html.escape(str(value))


def chart(title, series, x_label, y_label):
    width, height = 780, 300
    left, top, pw, ph = 70, 24, 680, 215
    xs = [x for _, points in series for x, _ in points]
    ys = [y for _, points in series for _, y in points]
    xmax = max(xs) or 1
    ymax = max(ys) * 1.08 or 1
    out = [
        f'<figure><figcaption>{esc(title)}</figcaption><svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">'
    ]
    for i in range(5):
        value = ymax * i / 4
        y = top + ph * (1 - i / 4)
        out.append(
            f'<line x1="{left}" y1="{y}" x2="{left + pw}" y2="{y}" stroke="#d9dfe7"/><text x="{left - 10}" y="{y + 4}" text-anchor="end">{value:.2f}</text>'
        )
        x = left + pw * i / 4
        out.append(
            f'<text x="{x}" y="{top + ph + 20}" text-anchor="middle">{xmax * i / 4:.1f}</text>'
        )
    for i, (name, points) in enumerate(series):
        coords = " ".join(
            f"{left + pw * x / xmax},{top + ph * (1 - y / ymax)}" for x, y in points
        )
        color = COLORS[i % len(COLORS)]
        out.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{coords}"/>'
        )
        out.append(
            f'<text x="{left + 210 * i}" y="{height - 8}" fill="{color}">{esc(name)}</text>'
        )
    out.append(
        f'<text x="{left + pw / 2}" y="{height - 34}" text-anchor="middle">{esc(x_label)}</text><text x="{left}" y="14">{esc(y_label)}</text></svg></figure>'
    )
    return "".join(out)


def bars(title, rows, unit):
    maximum = max(v for _, v in rows) or 1
    out = [
        f'<figure><figcaption>{esc(title)}</figcaption><svg viewBox="0 0 780 {len(rows) * 42 + 24}" role="img" aria-label="{esc(title)}">'
    ]
    for i, (label, value) in enumerate(rows):
        y = i * 42 + 10
        out.append(
            f'<text x="210" y="{y + 17}" text-anchor="end">{esc(label)}</text><rect x="225" y="{y}" width="{420 * value / maximum}" height="26" rx="4" fill="{COLORS[i % len(COLORS)]}"/><text x="{235 + 420 * value / maximum}" y="{y + 18}">{value:.2f} {esc(unit)}</text>'
        )
    return "".join(out) + "</svg></figure>"


def main():
    from mps_report import render_mps

    body = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Seed: learning efficiency pilot</title><style>\nbody{margin:0;background:#f3f5f9;color:#1f2938;font:16px/1.65 system-ui,sans-serif}main{max-width:1080px;margin:auto;padding:36px 24px}h1{font-size:44px;line-height:1.1;max-width:850px}h2{margin-top:40px}p{max-width:960px}.lead{font-size:20px}.tag{color:#3867e8;font-weight:700}figure,.card{background:white;padding:22px;border-radius:14px;margin:20px 0;box-shadow:0 2px 12px #10204008}figcaption{font-weight:700;margin-bottom:10px}svg{width:100%;height:auto}svg text{font:12px system-ui}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;text-align:right;border-bottom:1px solid #dce1ea}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}.muted{color:#596579}a{color:#2855c3}pre{white-space:pre-wrap;font-size:12px}details{margin:20px 0}code{font-size:14px}footer{margin-top:40px;border-top:1px solid #ccd4df;padding-top:16px}</style><main>\n<p class="tag">SEED RESEARCH · CPU + MPS STUDY · THREE PAIRED SEEDS</p><h1>Which small model learns more efficiently?</h1>'
    ]
    section, payload = render_mps(ROOT, chart, bars, esc)
    body.append(section)
    payload["corpora"] = json.loads((ROOT / "mps/corpora.json").read_text())
    body.append(
        f"<details><summary>Embedded measured data</summary><pre>{esc(json.dumps(payload, indent=2))}</pre></details>"
    )
    body.append(
        "<footer>Local Seed experiment · No network dependencies, external fonts, or chart libraries. Report generated from recorded results.</footer></main></html>"
    )
    (ROOT / "report.html").write_text("".join(body))


if __name__ == "__main__":
    main()
