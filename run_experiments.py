#!/usr/bin/env python3
"""Полный воспроизводимый эксперимент ЛР-4.

Запуск: ``python3 run_experiments.py``.  Время обычно менее минуты на
современном ноутбуке; сторонние библиотеки не требуются.
"""

from __future__ import annotations

import csv
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from direct_ga import run_direct_ga
from gp_hyperheuristic import run_gp
from jssp import dispatch, load_instances, normalized_makespan, rule_priority


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def save_csv(name: str, rows: List[Dict]) -> None:
    if not rows:
        return
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def mean(xs):
    values = list(xs)
    return sum(values) / len(values)


def stats(rows: List[Dict], method: str, metric: str) -> Dict:
    values = [float(r[metric]) for r in rows]
    return {"method": method, "metric": metric, "n": len(values), "best": min(values),
            "mean": mean(values), "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0, "worst": max(values)}


def svg_line_chart(path: Path, series: Dict[str, List[float]], title: str, y_label: str) -> None:
    width, height, left, right, top, bottom = 960, 460, 80, 25, 55, 60
    all_y = [v for ys in series.values() for v in ys]
    lo, hi = min(all_y), max(all_y)
    padding = max((hi - lo) * .08, .001); lo -= padding; hi += padding
    plot_w, plot_h = width-left-right, height-top-bottom
    def xy(i, y, n):
        x = left + (i / max(n-1, 1)) * plot_w
        yy = top + (hi-y)/(hi-lo)*plot_h
        return x, yy
    colors = ["#0868ac", "#d95f0e", "#238b45", "#756bb1"]
    text = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<style>text{font-family:Arial,sans-serif;fill:#17212b}.grid{stroke:#d9e1e8;stroke-width:1}.axis{stroke:#34495e;stroke-width:1.3}</style>',
            f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18" font-weight="bold">{html.escape(title)}</text>']
    for k in range(6):
        y = top + k*plot_h/5; val = hi-k*(hi-lo)/5
        text += [f'<line class="grid" x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}"/>', f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="11">{val:.3f}</text>']
    text += [f'<line class="axis" x1="{left}" x2="{width-right}" y1="{height-bottom}" y2="{height-bottom}"/>', f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{height-bottom}"/>', f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle" font-size="13">{html.escape(y_label)}</text>', f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-size="13">Поколение</text>']
    for color, (name, values) in zip(colors, series.items()):
        points = " ".join(f"{x:.1f},{y:.1f}" for i, v in enumerate(values) for x, y in [xy(i, v, len(values))])
        text += [f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>', f'<rect x="{width-235}" y="{58 + list(series).index(name)*23}" width="14" height="14" fill="{color}"/>', f'<text x="{width-216}" y="{70 + list(series).index(name)*23}" font-size="12">{html.escape(name)}</text>']
    text.append("</svg>")
    path.write_text("\n".join(text), encoding="utf-8")


def gantt_svg(path: Path, schedule: Dict, title: str) -> None:
    ops, ms = schedule["operations"], schedule["makespan"]
    m_count = max(op["machine"] for op in ops) + 1
    width, left, top, row_h = 1100, 110, 55, 35
    height = top + m_count*row_h + 60
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2", "#72b7b2", "#ff9da6", "#9d755d", "#bab0ac", "#59a14f"]
    text = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><style>text{{font-family:Arial,sans-serif;fill:#17212b}}</style>', f'<text x="{width/2}" y="25" text-anchor="middle" font-size="17" font-weight="bold">{html.escape(title)}</text>']
    plot_w = width-left-35
    for m in range(m_count):
        y = top+m*row_h
        text += [f'<rect x="{left}" y="{y}" width="{plot_w}" height="{row_h-4}" fill="#f5f7f9"/>', f'<text x="{left-8}" y="{y+20}" text-anchor="end" font-size="12">M{m+1}</text>']
    for op in ops:
        x, w = left + plot_w*op["start"]/ms, max(1, plot_w*op["duration"]/ms)
        y = top + op["machine"]*row_h
        text += [f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="{row_h-4}" fill="{colors[op["job"]%len(colors)]}" stroke="white"/>']
        if w > 28: text.append(f'<text x="{x+w/2:.2f}" y="{y+20}" text-anchor="middle" font-size="10" fill="white">J{op["job"]+1}</text>')
    text += [f'<text x="{left}" y="{height-18}" font-size="12">0</text>', f'<text x="{left+plot_w}" y="{height-18}" text-anchor="end" font-size="12">Cmax = {ms:.0f}</text>', "</svg>"]
    path.write_text("\n".join(text), encoding="utf-8")


def tree_html(path: Path, tree: Dict, expression: str) -> None:
    payload = json.dumps(tree, ensure_ascii=False)
    path.write_text(f"""<!doctype html><meta charset=\"utf-8\"><title>GP rule tree</title>
<style>body{{font:15px Arial;margin:28px;background:#f7f9fc;color:#17212b}}ul{{padding-left:26px}}li{{margin:7px}}.function{{color:#005a9c;font-weight:bold}}.variable{{color:#238b45}}.constant{{color:#b35806}}code{{background:#e9eef4;padding:8px;display:block;white-space:pre-wrap}}</style>
<h1>Лучшее эволюционировавшее правило</h1><code>{html.escape(expression)}</code><div id=tree></div>
<script>const root={payload};function draw(n){{let li=document.createElement('li');li.className=n.op_type;li.textContent=n.value;if(n.children.length){{let u=document.createElement('ul');n.children.forEach(x=>u.append(draw(x)));li.append(u)}}return li}}let u=document.createElement('ul');u.append(draw(root));document.querySelector('#tree').append(u);</script>""", encoding="utf-8")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    instances = load_instances(ROOT / "data/jssp_instances.json")
    train = [instances[x] for x in cfg["training_instances"]]
    test_names = cfg["test_instances"]
    test = [instances[x] for x in test_names]
    seeds = list(range(cfg["seed_start"], cfg["seed_start"] + cfg["num_independent_runs"]))
    rows, trajectories = [], defaultdict(list)
    best_overall = None
    for variant, ablated in (("GP: контроль разрастания", False), ("GP: абляция контроля", True)):
        all_histories = []
        for seed in seeds:
            result = run_gp(train, cfg["gp"], seed, ablated=ablated)
            test_scores = [normalized_makespan(ins, dispatch(ins, result["tree"].evaluate)["makespan"]) for ins in test]
            row = {"method": variant, "seed": seed, "train_normalized_makespan": result["train_loss"], "test_normalized_makespan": mean(test_scores), "tree_size": result["size"], "tree_depth": result["depth"], "fitness_evaluations": result["fitness_evaluations"], "expression": result["tree"].to_infix()}
            rows.append(row); all_histories.append(result["history"])
            if not ablated and (best_overall is None or row["test_normalized_makespan"] < best_overall[0]): best_overall = (row["test_normalized_makespan"], result)
        for g in range(cfg["gp"]["generations"] + 1):
            trajectories[variant].append(mean(h[g]["best_train"] for h in all_histories))
            trajectories[variant + " — размер"].append(mean(h[g]["mean_size"] for h in all_histories))
    # Прямой ГА — предметный ориентир, оптимизирующий каждый тестовый инстанс отдельно.
    direct_rows = []
    for name, ins in zip(test_names, test):
        for seed in seeds:
            r = run_direct_ga(ins, cfg["direct_ga"], seed)
            direct_rows.append({"method": "Прямой ГА", "instance": name, "seed": seed, "makespan": r["makespan"], "normalized_makespan": normalized_makespan(ins, r["makespan"]), "fitness_evaluations": r["fitness_evaluations"]})
    # Детерминированные ручные эвристики.
    baseline_rows = []
    for name, ins in zip(test_names, test):
        for rule in ("SPT", "LPT", "MWKR", "SLACK", "FIFO"):
            sched = dispatch(ins, rule_priority(rule))
            baseline_rows.append({"method": rule, "instance": name, "makespan": sched["makespan"], "normalized_makespan": normalized_makespan(ins, sched["makespan"])})
    save_csv("gp_runs.csv", rows); save_csv("direct_ga_runs.csv", direct_rows); save_csv("baseline_rules.csv", baseline_rows)
    summary = [stats([r for r in rows if r["method"] == m], m, "test_normalized_makespan") for m in ("GP: контроль разрастания", "GP: абляция контроля")]
    summary += [stats([r for r in direct_rows if r["instance"] == name], f"Прямой ГА ({name})", "normalized_makespan") for name in test_names]
    save_csv("summary.csv", summary)
    svg_line_chart(OUT / "convergence.svg", {k: v for k, v in trajectories.items() if "размер" not in k}, "Средняя лучшая обучающая пригодность (30 запусков)", "Нормированный makespan")
    svg_line_chart(OUT / "tree_size.svg", {k.replace(" — размер", ""): v for k, v in trajectories.items() if "размер" in k}, "Разрастание деревьев GP", "Средний размер дерева, узлы")
    score, best = best_overall
    best_tests = [(normalized_makespan(ins, dispatch(ins, best["tree"].evaluate)["makespan"]), name, ins) for name, ins in zip(test_names, test)]
    _, name, instance = min(best_tests)
    schedule = dispatch(instance, best["tree"].evaluate)
    gantt_svg(OUT / "best_schedule_gantt.svg", schedule, f"Расписание лучшего GP-правила: {name}")
    (OUT / "best_rule.json").write_text(json.dumps({"expression": best["tree"].to_infix(), "tree": best["tree"].to_dict(), "training_score": best["train_loss"], "test_score": score}, ensure_ascii=False, indent=2), encoding="utf-8")
    tree_html(OUT / "best_rule_tree.html", best["tree"].to_dict(), best["tree"].to_infix())
    info = {"seeds": seeds, "dataset_seed": cfg["dataset_seed"], "gp_candidate_schedule_evaluations_per_run": cfg["gp"]["population_size"] * (cfg["gp"]["generations"] + 1) * len(train), "direct_ga_candidate_schedule_evaluations_per_run": cfg["direct_ga"]["population_size"] * (cfg["direct_ga"]["generations"] + 1), "comparison_note": "Два варианта GP используют абсолютно одинаковый бюджет; прямой ГА служит отдельным специализированным ориентиром на каждом тестовом инстансе."}
    (OUT / "run_information.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Done. Results:", OUT)


if __name__ == "__main__":
    main()
