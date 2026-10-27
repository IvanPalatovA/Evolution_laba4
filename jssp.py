"""Модель задачи Job Shop Scheduling и детерминированные диспетчеры.

В файле намеренно нет зависимостей от сторонних пакетов: это делает
эксперимент из лабораторной работы воспроизводимым на обычном Python 3.10+.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple


Instance = Dict


def load_instances(path: str | Path = "data/jssp_instances.json") -> Dict[str, Instance]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    result: Dict[str, Instance] = {}
    for section in ("benchmarks", "training_suite", "test_suite"):
        result.update(data[section])
    return result


def job_due_dates(instance: Instance) -> List[float]:
    """Искусственные сроки нужны лишь как входной признак правила, не как цель."""
    return [1.5 * sum(op["duration"] for op in job) for job in instance["jobs"]]


def _features(instance: Instance, state: Dict, job: int, now: float) -> Dict[str, float]:
    jobs = instance["jobs"]
    op_index = state["next_op"][job]
    op = jobs[job][op_index]
    machine = op["machine"]
    p = float(op["duration"])
    w = float(sum(x["duration"] for x in jobs[job][op_index:]))
    waiting = max(0.0, now - state["job_ready"][job])
    queued = [
        j for j, idx in enumerate(state["next_op"])
        if idx < len(jobs[j]) and jobs[j][idx]["machine"] == machine
        and state["job_ready"][j] <= now
    ]
    q_work = sum(jobs[j][state["next_op"][j]]["duration"] for j in queued)
    dd = state["due_dates"][job]
    return {
        "p": p, "w": w, "rem_ops": float(len(jobs[job]) - op_index),
        "wait": waiting, "t": now, "dd": dd, "slack": dd - now - w,
        "q_len": float(len(queued)), "q_work": float(q_work),
    }


def dispatch(instance: Instance, priority: Callable[[Dict[str, float]], float],
             prefer_low: bool = True, rng: random.Random | None = None) -> Dict:
    """Строит допустимое активное расписание по правилу приоритета.

    На каждом шаге отбираются операции с наименьшим достижимым временем старта;
    правило разрешает только конфликт между ними. Поэтому все precedence- и
    machine-ограничения соблюдаются по построению.
    """
    jobs = instance["jobs"]
    machines = instance["metrics"]["num_machines"]
    state = {
        "next_op": [0] * len(jobs), "job_ready": [0.0] * len(jobs),
        "machine_ready": [0.0] * machines, "due_dates": job_due_dates(instance),
    }
    operations: List[Dict] = []
    total = sum(len(j) for j in jobs)
    while len(operations) < total:
        feasible: List[Tuple[float, int]] = []
        for j, idx in enumerate(state["next_op"]):
            if idx < len(jobs[j]):
                m = jobs[j][idx]["machine"]
                feasible.append((max(state["job_ready"][j], state["machine_ready"][m]), j))
        now = min(x[0] for x in feasible)
        candidates = [j for start, j in feasible if start == now]
        scores = []
        for j in candidates:
            try:
                value = float(priority(_features(instance, state, j, now)))
            except (ArithmeticError, ValueError, TypeError):
                value = 0.0
            if value != value or abs(value) > 1e15:
                value = 0.0
            scores.append((value, j))
        target = min(x[0] for x in scores) if prefer_low else max(x[0] for x in scores)
        tied = [j for score, j in scores if abs(score - target) < 1e-12]
        chosen = rng.choice(tied) if rng and len(tied) > 1 else min(tied)
        op_index = state["next_op"][chosen]
        op = jobs[chosen][op_index]
        machine = op["machine"]
        start = max(state["job_ready"][chosen], state["machine_ready"][machine])
        end = start + op["duration"]
        operations.append({"job": chosen, "operation": op_index, "machine": machine,
                           "start": start, "end": end, "duration": op["duration"]})
        state["next_op"][chosen] += 1
        state["job_ready"][chosen] = end
        state["machine_ready"][machine] = end
    return {"makespan": max(state["machine_ready"]), "operations": operations}


def rule_priority(name: str) -> Callable[[Dict[str, float]], float]:
    """Набор прозрачных ручных baseline-эвристик."""
    if name == "SPT":
        return lambda x: x["p"]
    if name == "LPT":
        return lambda x: -x["p"]
    if name == "MWKR":
        return lambda x: -x["w"]
    if name == "SLACK":
        return lambda x: x["slack"]
    if name == "FIFO":
        return lambda x: -x["wait"]
    raise ValueError(f"Unknown baseline rule: {name}")


def normalized_makespan(instance: Instance, makespan: float) -> float:
    return makespan / instance["metrics"]["lower_bound_makespan"]
