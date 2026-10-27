"""Генетическое программирование диспетчерских правил для JSSP."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from gp_tree import Node, hoist_mutation, point_mutation, ramped_half_and_half, subtree_crossover, subtree_mutation
from jssp import dispatch, normalized_makespan


@dataclass
class Individual:
    tree: Node
    loss: float = float("inf")
    raw_loss: float = float("inf")


def evaluate(tree: Node, instances: Iterable[Dict], parsimony_weight: float) -> Tuple[float, float]:
    values = [normalized_makespan(inst, dispatch(inst, tree.evaluate)["makespan"]) for inst in instances]
    raw = sum(values) / len(values)
    return raw + parsimony_weight * tree.size(), raw


def tournament(rng: random.Random, population: List[Individual], size: int) -> Individual:
    return min((rng.choice(population) for _ in range(size)), key=lambda x: x.loss)


def run_gp(instances: List[Dict], config: Dict, seed: int, ablated: bool = False) -> Dict:
    """Один прогон GP. У ablated отключены hoist и штраф сложности."""
    rng = random.Random(seed)
    pop_size, generations = config["population_size"], config["generations"]
    parsimony = 0.0 if ablated else config["parsimony_weight"]
    population = [Individual(t) for t in ramped_half_and_half(rng, pop_size, 1, config["max_init_depth"])]
    history: List[Dict] = []
    evaluations = 0
    for generation in range(generations + 1):
        for ind in population:
            ind.loss, ind.raw_loss = evaluate(ind.tree, instances, parsimony)
        evaluations += len(population) * len(instances)
        population.sort(key=lambda x: x.loss)
        mean_raw = sum(i.raw_loss for i in population) / len(population)
        history.append({"generation": generation, "best_train": population[0].raw_loss,
                        "mean_train": mean_raw, "best_size": population[0].tree.size(),
                        "mean_size": sum(i.tree.size() for i in population) / len(population)})
        if generation == generations:
            break
        offspring = [Individual(population[i].tree.clone()) for i in range(config["elitism"])]
        while len(offspring) < pop_size:
            a = tournament(rng, population, config["tournament_size"])
            b = tournament(rng, population, config["tournament_size"])
            if rng.random() < config["crossover_prob"]:
                c1, c2 = subtree_crossover(rng, a.tree, b.tree, config["max_depth"])
            else:
                c1, c2 = a.tree.clone(), b.tree.clone()
            for child in (c1, c2):
                r = rng.random()
                if r < config["subtree_mutation_prob"]:
                    child = subtree_mutation(rng, child, config["max_depth"])
                elif r < config["subtree_mutation_prob"] + config["point_mutation_prob"]:
                    child = point_mutation(rng, child)
                elif not ablated and r < (config["subtree_mutation_prob"] + config["point_mutation_prob"] + config["hoist_mutation_prob"]):
                    child = hoist_mutation(rng, child)
                offspring.append(Individual(child))
                if len(offspring) >= pop_size:
                    break
        population = offspring
    best = min(population, key=lambda x: x.loss)
    return {"tree": best.tree.clone(), "train_loss": best.raw_loss, "size": best.tree.size(),
            "depth": best.tree.depth(), "history": history, "fitness_evaluations": evaluations}
