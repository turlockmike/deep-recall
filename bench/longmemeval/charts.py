#!/usr/bin/env python3
"""Render the README charts (static SVG, light and dark) from the committed LongMemEval results.

  python bench/longmemeval/charts.py            # writes docs/img/reranker-lift.svg, docs/img/longmemeval-misses.svg

deep-recall numbers come from bench/longmemeval/results/*/summary.json. Numbers for other systems are the
constants in OTHERS, each with the source it was read from.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "docs" / "img"

# Other systems: session-level recall_any@5 on all 500 LongMemEval-S questions, as each project publishes it.
OTHERS = [
    ("MemPalace (raw)", 0.966, "https://github.com/MemPalace/mempalace (README; recall_any@5 in benchmarks/longmemeval_bench.py)"),
    ("agentmemory", 0.952, "https://github.com/rohitg00/agentmemory/blob/main/benchmark/LONGMEMEVAL.md (recall_any@K)"),
]

LIGHT = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
         "axis": "#c3c2b7", "gray": "#a8a69e", "accent": "#2a78d6"}
DARK = {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
        "axis": "#383835", "gray": "#6b6a65", "accent": "#3987e5"}
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"


def _style() -> str:
    def block(t):
        return (f".bg{{fill:{t['surface']}}}.ink{{fill:{t['ink']}}}.ink2{{fill:{t['ink2']}}}.muted{{fill:{t['muted']}}}"
                f".grid{{stroke:{t['grid']}}}.axis{{stroke:{t['axis']}}}.gray{{fill:{t['gray']}}}.accent{{fill:{t['accent']}}}")
    return f"<style>text{{font-family:{FONT}}}{block(LIGHT)}@media (prefers-color-scheme:dark){{{block(DARK)}}}</style>"


def _bar(x: float, y: float, w: float, h: float, cls: str) -> str:
    """A horizontal bar: square at the baseline, 4px rounded data-end."""
    r = min(4.0, w / 2, h / 2)
    if w <= 0:
        return ""
    return (f'<path class="{cls}" d="M{x:.1f},{y:.1f}h{w - r:.1f}a{r},{r} 0 0 1 {r},{r}v{h - 2 * r:.1f}'
            f'a{r},{r} 0 0 1 -{r},{r}h-{w - r:.1f}z"/>')


def _svg(width: int, height: int, title: str, desc: str, body: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'role="img" aria-labelledby="t d"><title id="t">{title}</title><desc id="d">{desc}</desc>{_style()}'
            f'<rect class="bg" width="{width}" height="{height}" rx="8"/>{body}</svg>\n')


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%" if v * 100 % 1 else f"{v * 100:.0f}%"


def reranker_lift(lme: dict, lme500: dict) -> str:
    """Paired bars per LongMemEval-S measure: hybrid search vs the same search plus the answerability reranker."""
    W, x0, plot = 760, 250, 420
    groups = [
        ("Every evidence session in the top 5", f"LongMemEval-S · {lme['n']} answerable questions",
         lme["search"]["all@5"] / lme["n"], lme["recall"]["all@5"] / lme["n"], ""),
        ("Any evidence session in the top 5", f"LongMemEval-S · all {lme500['n']} questions",
         lme500["search"]["any@5"] / lme500["n"], lme500["recall"]["any@5"] / lme500["n"], ""),
    ]
    body = [f'<text class="ink" x="24" y="40" font-size="20" font-weight="600">Search finds notes about the question. '
            f'The reranker finds the answer.</text>',
            f'<text class="ink2" x="24" y="64" font-size="13">Blue: the same search plus deep-recall\'s rare-term '
            f'candidates, ranked by its answerability reranker (jev).</text>']
    y = 96
    for name, metric, s, r, approx in groups:
        body.append(f'<text class="ink" x="24" y="{y + 14}" font-size="14" font-weight="600">{name}'
                    f'<tspan class="muted" font-size="12" font-weight="400" dx="10">{metric}</tspan></text>')
        y += 26
        for i, (lab, v, cls, pre) in enumerate([("Hybrid search", s, "gray", ""), ("+ reranker", r, "accent", approx)]):
            by = y + i * 30
            body.append(f'<text class="ink2" x="{x0 - 10}" y="{by + 15}" font-size="12" text-anchor="end">{lab}</text>')
            body.append(_bar(x0, by, plot * v, 22, cls))
            weight = ' font-weight="700"' if cls == "accent" else ""
            body.append(f'<text class="ink" x="{x0 + plot * v + 8}" y="{by + 16}" font-size="14"{weight}>{pre}{_pct(v)}</text>')
        body.append(f'<line class="axis" x1="{x0}" x2="{x0}" y1="{y - 4}" y2="{y + 56}" stroke-width="1"/>')
        y += 80
    body.append(f'<text class="muted" x="24" y="{y + 4}" font-size="11">Retrieval, not answer accuracy: whether the '
                f'context handed to your LLM holds the facts.</text>')
    body.append(f'<text class="muted" x="24" y="{y + 20}" font-size="11">LongMemEval: github.com/xiaowu0162/LongMemEval'
                f' · results and method: bench/longmemeval/</text>')
    return _svg(W, y + 38, "What the reranker adds",
                "Hybrid search alone versus with deep-recall's answerability reranker on LongMemEval-S. "
                f"Every evidence session in the top 5: {_pct(groups[0][2])} to {_pct(groups[0][3])}. "
                f"Any evidence session in the top 5: {_pct(groups[1][2])} to {_pct(groups[1][3])}.", "".join(body))


def misses(lme500: dict) -> str:
    """Questions (of 500) where no evidence session reached the top 5; lower is better."""
    n = lme500["n"]
    rows = [(name, round(n * (1 - v)), "gray", src) for name, v, src in OTHERS]
    rows.append(("deep-recall, search only", n - lme500["search"]["any@5"], "gray", ""))
    rows.sort(key=lambda r: -r[1])
    rows.append(("deep-recall + reranker", n - lme500["recall"]["any@5"], "accent", ""))
    W, x0, plot = 760, 230, 440
    top = max(r[1] for r in rows)
    body = [f'<text class="ink" x="24" y="40" font-size="20" font-weight="600">Questions where the answer never '
            f'reached the context</text>',
            f'<text class="ink2" x="24" y="64" font-size="13">LongMemEval-S, all {n} questions: no evidence session in '
            f'the top 5 retrieved. Fewer is better.</text>']
    y = 88
    for name, miss, cls, _ in rows:
        weight = ' font-weight="700"' if cls == "accent" else ""
        body.append(f'<text class="ink2" x="{x0 - 10}" y="{y + 16}" font-size="13" text-anchor="end"{weight}>{name}</text>')
        body.append(_bar(x0, y, max(plot * miss / top, 3), 22, cls))
        body.append(f'<text class="ink" x="{x0 + max(plot * miss / top, 3) + 8}" y="{y + 16}" font-size="14"{weight}>'
                    f'{miss} of {n}</text>')
        y += 34
    body.append(f'<line class="axis" x1="{x0}" x2="{x0}" y1="84" y2="{y - 8}" stroke-width="1"/>')
    body.append(f'<text class="muted" x="24" y="{y + 10}" font-size="11">Session-level recall_any@5 as each project '
                f'publishes it: MemPalace README, agentmemory benchmark/LONGMEMEVAL.md.</text>')
    body.append(f'<text class="muted" x="24" y="{y + 26}" font-size="11">deep-recall: bench/longmemeval/results/'
                f's-500-contrib-prompt.</text>')
    return _svg(W, y + 44, "Questions where the answer never reached the context",
                "; ".join(f"{r[0]}: {r[1]} of {n}" for r in rows), "".join(body))


def main() -> None:
    res = HERE / "results"
    lme = json.loads((res / "s-470-contrib-prompt" / "summary.json").read_text())
    lme500 = json.loads((res / "s-500-contrib-prompt" / "summary.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reranker-lift.svg").write_text(reranker_lift(lme, lme500))
    (OUT / "longmemeval-misses.svg").write_text(misses(lme500))
    print(json.dumps({"wrote": [str(p.relative_to(HERE.parent.parent)) for p in sorted(OUT.glob("*.svg"))]}))


if __name__ == "__main__":
    main()
