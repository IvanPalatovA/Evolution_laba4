#!/usr/bin/env python3
"""
generate_data.py
Генератор и сборщик набора данных для лабораторной работы № 4.
Включает классические мировые бенчмарки (FT06, FT10, LA01, LA02)
и генерацию детерминированного набора синтетических обучающих и тестовых
инстансов задачи календарного планирования (Job Shop Scheduling Problem, JSSP).
Все синтетические инстансы воспроизводятся детерминированно с фиксированным seed.
"""

import json
import os
import random
from pathlib import Path


# 1. Классические бенчмарки из OR-Library
RAW_BENCHMARKS = {
    "FT06": {
        "description": "Fisher & Thompson 6x6 benchmark (optimal makespan: 55)",
        "num_jobs": 6,
        "num_machines": 6,
        "optimal_makespan": 55,
        "raw": """
        2 1 0 3 1 6 3 7 5 3 4 6
        1 8 2 5 4 10 5 10 0 10 3 4
        2 5 3 4 5 8 0 9 1 1 4 7
        1 5 0 5 2 5 3 3 4 8 5 9
        2 9 1 3 4 5 5 4 0 3 3 1
        1 3 3 3 5 9 0 10 4 4 2 1
        """
    },
    "FT10": {
        "description": "Fisher & Thompson 10x10 benchmark (optimal makespan: 930)",
        "num_jobs": 10,
        "num_machines": 10,
        "optimal_makespan": 930,
        "raw": """
        0 29 1 78 2  9 3 36 4 49 5 11 6 62 7 56 8 44 9 21
        0 43 2 90 4 75 9 11 3 69 1 28 6 46 5 46 7 72 8 30
        1 91 0 85 3 39 2 74 8 90 5 10 7 12 6 89 9 45 4 33
        1 81 2 95 0 71 4 99 6  9 8 52 7 85 3 98 9 22 5 43
        2 14 0  6 1 22 5 61 3 26 4 69 8 21 7 49 9 72 6 53
        2 84 1  2 5 52 3 95 8 48 9 72 0 47 6 65 4  6 7 25
        1 46 0 37 3 61 2 13 6 32 5 21 9 32 8 89 7 30 4 55
        2 31 0 86 1 46 5 74 4 32 6 88 8 19 9 48 7 36 3 79
        0 76 1 69 3 76 5 51 2 85 9 11 6 40 7 89 4 26 8 74
        1 85 0 13 2 61 6  7 8 64 9 76 5 47 3 52 4 90 7 45
        """
    },
    "LA01": {
        "description": "Lawrence 10x5 benchmark (optimal makespan: 666)",
        "num_jobs": 10,
        "num_machines": 5,
        "optimal_makespan": 666,
        "raw": """
        1 21 0 53 4 95 3 55 2 34
        0 21 3 52 4 16 2 26 1 71
        3 39 4 98 1 42 2 31 0 12
        1 77 0 55 4 79 2 66 3 77
        0 83 3 34 2 64 1 19 4 37
        1 54 2 43 4 79 0 92 3 62
        3 69 4 77 1 87 2 87 0 93
        2 38 0 60 1 41 3 24 4 83
        3 17 1 49 4 25 0 44 2 98
        4 77 3 79 2 43 1 75 0 96
        """
    },
    "LA02": {
        "description": "Lawrence 10x5 benchmark (optimal makespan: 655)",
        "num_jobs": 10,
        "num_machines": 5,
        "optimal_makespan": 655,
        "raw": """
        0 20 3 87 1 31 4 76 2 17
        4 25 2 32 0 24 1 18 3 81
        1 72 2 23 4 28 0 58 3 99
        2 86 1 76 4 97 0 45 3 90
        4 27 0 42 3 48 2 17 1 46
        1 67 0 98 4 48 3 27 2 62
        4 28 1 12 3 19 0 80 2 50
        1 63 0 94 2 98 3 50 4 80
        4 14 0 75 2 50 1 41 3 55
        4 72 2 18 1 37 3 79 0 61
        """
    }
}


def parse_raw_instance(raw_text: str):
    """Преобразует строковое представление инстанса в список операций."""
    lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip()]
    jobs = []
    for line in lines:
        nums = list(map(int, line.split()))
        ops = []
        for k in range(0, len(nums), 2):
            ops.append({"machine": nums[k], "duration": nums[k + 1]})
        jobs.append(ops)
    return jobs


def compute_instance_metrics(jobs, num_machines):
    """Вычисляет базовые метрики и нижнюю оценку makespan (Lower Bound)."""
    num_jobs = len(jobs)
    job_work = [sum(op["duration"] for op in job) for job in jobs]
    machine_work = [0] * num_machines
    for job in jobs:
        for op in job:
            machine_work[op["machine"]] += op["duration"]
    
    max_job_work = max(job_work) if job_work else 0
    max_machine_work = max(machine_work) if machine_work else 0
    lower_bound = max(max_job_work, max_machine_work)
    total_ops = sum(len(job) for job in jobs)
    
    return {
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "total_operations": total_ops,
        "max_job_duration": max_job_work,
        "max_machine_load": max_machine_work,
        "lower_bound_makespan": lower_bound,
        "total_workload": sum(job_work)
    }


def generate_synthetic_instance(rng: random.Random, num_jobs: int, num_machines: int, min_dur=10, max_dur=99):
    """
    Генерирует воспроизводимый инстанс JSSP заданного размера.
    Каждая работа проходит через все num_machines в случайном уникальном порядке.
    """
    jobs = []
    for _ in range(num_jobs):
        machine_order = list(range(num_machines))
        rng.shuffle(machine_order)
        job_ops = []
        for m in machine_order:
            dur = rng.randint(min_dur, max_dur)
            job_ops.append({"machine": m, "duration": dur})
        jobs.append(job_ops)
    return jobs


def generate_all_datasets(seed=2026, output_file="data/jssp_instances.json"):
    """Создает полный каталог задач: бенчмарки, обучающую выборку и тестовую выборку."""
    rng = random.Random(seed)
    dataset = {
        "metadata": {
            "generator_seed": seed,
            "created_for": "Laboratory Work 4 - Genetic Programming of Heuristics",
            "problem": "Job Shop Scheduling Problem (JSSP)"
        },
        "benchmarks": {},
        "training_suite": {},
        "test_suite": {}
    }

    # 1. Классические бенчмарки
    for name, info in RAW_BENCHMARKS.items():
        jobs = parse_raw_instance(info["raw"])
        metrics = compute_instance_metrics(jobs, info["num_machines"])
        metrics["optimal_makespan"] = info.get("optimal_makespan")
        dataset["benchmarks"][name] = {
            "description": info["description"],
            "metrics": metrics,
            "jobs": jobs
        }

    # 2. Обучающая выборка (8 разнообразных инстансов)
    train_configs = [
        ("synth_train_1", 6, 6),
        ("synth_train_2", 6, 6),
        ("synth_train_3", 8, 6),
        ("synth_train_4", 8, 6),
        ("synth_train_5", 8, 8),
        ("synth_train_6", 8, 8),
        ("synth_train_7", 10, 8),
        ("synth_train_8", 10, 8),
    ]
    for name, n_j, n_m in train_configs:
        jobs = generate_synthetic_instance(rng, n_j, n_m)
        metrics = compute_instance_metrics(jobs, n_m)
        dataset["training_suite"][name] = {
            "description": f"Synthetic training instance {n_j} jobs x {n_m} machines",
            "metrics": metrics,
            "jobs": jobs
        }

    # 3. Крупные масштабируемые тестовые инстансы (обобщающая способность)
    test_configs = [
        ("synth_test_large_1", 15, 10),
        ("synth_test_large_2", 20, 10),
    ]
    for name, n_j, n_m in test_configs:
        jobs = generate_synthetic_instance(rng, n_j, n_m)
        metrics = compute_instance_metrics(jobs, n_m)
        dataset["test_suite"][name] = {
            "description": f"Large-scale test instance {n_j} jobs x {n_m} machines (generalization)",
            "metrics": metrics,
            "jobs": jobs
        }

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Dataset successfully generated at {out_path}")
    print(f"Total benchmarks: {len(dataset['benchmarks'])}")
    print(f"Total training instances: {len(dataset['training_suite'])}")
    print(f"Total large test instances: {len(dataset['test_suite'])}")
    for category in ["benchmarks", "training_suite", "test_suite"]:
        print(f"\n--- {category.upper()} ---")
        for k, v in dataset[category].items():
            m = v["metrics"]
            opt = f", optimal={m.get('optimal_makespan')}" if m.get("optimal_makespan") else ""
            print(f"  {k:18}: {m['num_jobs']}x{m['num_machines']} jobs/machines, "
                  f"{m['total_operations']} ops, LB makespan={m['lower_bound_makespan']}{opt}")

    return dataset


if __name__ == "__main__":
    generate_all_datasets()

