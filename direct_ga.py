"""Прямой ГА для JSSP: хромосома — последовательность идентификаторов работ."""

from __future__ import annotations

import random
from typing import Dict, List


def decode(instance: Dict, chromosome: List[int]) -> Dict:
    jobs = instance["jobs"]
    machine_ready = [0] * instance["metrics"]["num_machines"]
    job_ready = [0] * len(jobs)
    next_op = [0] * len(jobs)
    operations = []
    for job in chromosome:
        idx = next_op[job]
        if idx >= len(jobs[job]):
            continue
        op = jobs[job][idx]
        start = max(job_ready[job], machine_ready[op["machine"]])
        end = start + op["duration"]
        operations.append({"job": job, "operation": idx, "machine": op["machine"], "start": start, "end": end, "duration": op["duration"]})
        job_ready[job] = end
        machine_ready[op["machine"]] = end
        next_op[job] += 1
    return {"makespan": max(machine_ready), "operations": operations}


def _base(instance: Dict) -> List[int]:
    return [j for j, ops in enumerate(instance["jobs"]) for _ in ops]


def _crossover(rng: random.Random, a: List[int], b: List[int], capacities: List[int]) -> List[int]:
    cut = rng.randrange(len(a) + 1)
    child, counts = a[:cut], [0] * len(capacities)
    for x in child:
        counts[x] += 1
    for x in b + a:
        if counts[x] < capacities[x]:
            child.append(x); counts[x] += 1
        if len(child) == len(a):
            return child
    return child


def run_direct_ga(instance: Dict, config: Dict, seed: int) -> Dict:
    rng = random.Random(seed)
    base = _base(instance)
    capacities = [len(j) for j in instance["jobs"]]
    population = []
    for _ in range(config["population_size"]):
        x = base[:]; rng.shuffle(x); population.append(x)
    history = []
    evals = 0
    for generation in range(config["generations"] + 1):
        scored = [(decode(instance, x)["makespan"], x) for x in population]
        evals += len(scored)
        scored.sort(key=lambda x: x[0])
        history.append({"generation": generation, "best": scored[0][0],
                        "mean": sum(x[0] for x in scored) / len(scored)})
        if generation == config["generations"]:
            break
        def pick():
            return min((rng.choice(scored) for _ in range(config["tournament_size"])), key=lambda x: x[0])[1]
        population = [x[:] for _, x in scored[:config["elitism"]]]
        while len(population) < config["population_size"]:
            a, b = pick(), pick()
            child = _crossover(rng, a, b, capacities) if rng.random() < config["crossover_prob"] else a[:]
            if rng.random() < config["mutation_prob"]:
                i, j = rng.sample(range(len(child)), 2)
                child[i], child[j] = child[j], child[i]
            population.append(child)
    best_makespan, best = min((decode(instance, x)["makespan"], x) for x in population)
    return {"makespan": best_makespan, "chromosome": best, "schedule": decode(instance, best),
            "history": history, "fitness_evaluations": evals}
