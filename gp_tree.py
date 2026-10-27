#!/usr/bin/env python3
"""
gp_tree.py
Синтаксическое дерево выражений (Symbolic Expression Tree) для генетического
программирования эвристик (GP Hyper-Heuristics).
Реализует:
- Узлы деревьев: функции (бинарные, тернарные) и терминалы (переменные, константы);
- Защищенное исполнение (protected evaluation);
- Инициализацию Ramped Half-and-Half (методы full и grow);
- Генетические операторы: скрещивание поддеревьев (subtree crossover),
  мутацию поддерева (subtree mutation), точечную мутацию (point mutation),
  hoist-мутацию (сжатие разрастания);
- Вычисление глубины, размера, текстовое представление (инфиксное/префиксное)
  и экспорт в словарь/JSON для визуализации.
"""

import copy
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union


# Допустимые имена терминалов признаков задачи календарного планирования
FEATURE_TERMINALS = [
    "p",        # Длительность текущей операции
    "w",        # Оставшаяся трудоемкость всей работы (включая текущую операцию)
    "rem_ops",  # Число оставшихся операций работы
    "wait",     # Время ожидания работы в текущей очереди станка (t - ready_time)
    "t",        # Текущее время моделирования
    "dd",       # Директивный срок сдачи (due date) работы
    "slack",    # Резерв времени до директивного срока (dd - t - w)
    "q_len",    # Текущая длина очереди ожидания станка
    "q_work"    # Суммарная трудоемкость всех операций в очереди станка
]

# Константы (Ephemeral Random Constants)
CONSTANT_VALUES = [0.5, 1.0, 2.0]

# Функции и их арность
FUNCTION_ARITY = {
    "+": 2,
    "-": 2,
    "*": 2,
    "/": 2,
    "min": 2,
    "max": 2,
    "if_gt": 3   # if_gt(a, b, c) -> b if a > 0 else c
}


class Node:
    """Узел синтаксического дерева генетического программирования."""
    __slots__ = ("op_type", "value", "children")

    def __init__(self, op_type: str, value: Any, children: Optional[List["Node"]] = None):
        """
        op_type: 'function', 'variable', 'constant'
        value: имя функции ('+', 'min'...), имя переменной ('p', 'w'...) или числовое значение константы
        children: список дочерних узлов Node
        """
        self.op_type = op_type
        self.value = value
        self.children = children if children is not None else []

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def evaluate(self, env: Dict[str, float]) -> float:
        """Безопасное вычисление значения выражения для заданного словаря признаков."""
        if self.op_type == "constant":
            return float(self.value)
        elif self.op_type == "variable":
            val = env.get(self.value, 0.0)
            if math.isnan(val) or math.isinf(val):
                return 0.0
            return float(val)
        elif self.op_type == "function":
            fn = self.value
            if fn == "+":
                return self.children[0].evaluate(env) + self.children[1].evaluate(env)
            elif fn == "-":
                return self.children[0].evaluate(env) - self.children[1].evaluate(env)
            elif fn == "*":
                v1 = self.children[0].evaluate(env)
                v2 = self.children[1].evaluate(env)
                res = v1 * v2
                # Защита от переполнения
                if res > 1e12:
                    return 1e12
                elif res < -1e12:
                    return -1e12
                return res
            elif fn == "/":
                v1 = self.children[0].evaluate(env)
                v2 = self.children[1].evaluate(env)
                if abs(v2) < 1e-6:
                    return 1.0  # Защищенное деление Козы
                res = v1 / v2
                if res > 1e12:
                    return 1e12
                elif res < -1e12:
                    return -1e12
                return res
            elif fn == "min":
                return min(self.children[0].evaluate(env), self.children[1].evaluate(env))
            elif fn == "max":
                return max(self.children[0].evaluate(env), self.children[1].evaluate(env))
            elif fn == "if_gt":
                cond = self.children[0].evaluate(env)
                if cond > 0.0:
                    return self.children[1].evaluate(env)
                else:
                    return self.children[2].evaluate(env)
        return 0.0

    def size(self) -> int:
        """Число узлов в данном поддереве."""
        return 1 + sum(ch.size() for ch in self.children)

    def depth(self) -> int:
        """Глубина поддерева (глубина листа = 0)."""
        if self.is_leaf():
            return 0
        return 1 + max(ch.depth() for ch in self.children)

    def clone(self) -> "Node":
        """Глубокая копия поддерева."""
        return Node(
            self.op_type,
            self.value,
            [ch.clone() for ch in self.children]
        )

    def to_infix(self) -> str:
        """Инфиксное представление для чтения человеком."""
        if self.op_type == "constant":
            return f"{self.value:g}"
        elif self.op_type == "variable":
            return str(self.value)
        elif self.op_type == "function":
            fn = self.value
            if fn in ("+", "-", "*", "/"):
                left = self.children[0].to_infix()
                right = self.children[1].to_infix()
                return f"({left} {fn} {right})"
            elif fn in ("min", "max"):
                left = self.children[0].to_infix()
                right = self.children[1].to_infix()
                return f"{fn}({left}, {right})"
            elif fn == "if_gt":
                c = self.children[0].to_infix()
                t = self.children[1].to_infix()
                e = self.children[2].to_infix()
                return f"if({c} > 0, {t}, {e})"
        return "?"

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в структуру для JSON/HTML визуализатора."""
        return {
            "op_type": self.op_type,
            "value": str(self.value) if self.op_type != "constant" else self.value,
            "children": [ch.to_dict() for ch in self.children]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        """Десериализация из структуры словаря."""
        children = [cls.from_dict(c) for c in data.get("children", [])]
        return cls(data["op_type"], data["value"], children)

    def get_all_nodes(self) -> List["Node"]:
        """Возвращает список всех узлов поддерева в порядке прямого обхода."""
        res = [self]
        for ch in self.children:
            res.extend(ch.get_all_nodes())
        return res

    def get_all_subtrees_with_parent(self) -> List[Tuple["Node", Optional["Node"], int]]:
        """
        Возвращает список кортежей (node, parent, child_index).
        Для корня parent=None, child_index=-1.
        """
        items: List[Tuple["Node", Optional["Node"], int]] = [(self, None, -1)]
        queue = [(self, None, -1)]
        while queue:
            curr, p, idx = queue.pop(0)
            for i, ch in enumerate(curr.children):
                items.append((ch, curr, i))
                queue.append((ch, curr, i))
        return items


def make_random_leaf(rng: random.Random) -> Node:
    """Создает случайный лист: терминал-переменную либо константу."""
    if rng.random() < 0.75:
        var = rng.choice(FEATURE_TERMINALS)
        return Node("variable", var)
    else:
        c = rng.choice(CONSTANT_VALUES)
        return Node("constant", c)


def make_random_function_node(rng: random.Random) -> Node:
    """Создает функциональный узел со случайным оператором."""
    fn = rng.choice(list(FUNCTION_ARITY.keys()))
    return Node("function", fn)


def generate_tree(rng: random.Random, max_depth: int, method: str = "grow") -> Node:
    """
    Генерирует случайное дерево выражений.
    method: 'full' (все ветви доходят до max_depth) или 'grow' (ветви могут заканчиваться раньше).
    """
    if max_depth <= 0 or (method == "grow" and rng.random() < 0.25):
        return make_random_leaf(rng)

    # Выбираем функциональный узел
    fn = rng.choice(list(FUNCTION_ARITY.keys()))
    arity = FUNCTION_ARITY[fn]
    children = []
    for _ in range(arity):
        children.append(generate_tree(rng, max_depth - 1, method))
    return Node("function", fn, children)


def ramped_half_and_half(rng: random.Random, pop_size: int, min_depth: int = 1, max_depth: int = 4) -> List[Node]:
    """
    Классическая инициализация Ramped Half-and-Half (Дж. Коза).
    Половина популяции генерируется методом 'full', вторая половина - методом 'grow'
    с равномерным распределением глубин от min_depth до max_depth.
    """
    pop = []
    depths = list(range(min_depth, max_depth + 1))
    num_per_depth = pop_size // len(depths)
    
    for d in depths:
        for i in range(num_per_depth):
            method = "full" if (i % 2 == 0) else "grow"
            pop.append(generate_tree(rng, d, method))

    # Добиваем остаток до pop_size
    while len(pop) < pop_size:
        d = rng.choice(depths)
        method = rng.choice(["full", "grow"])
        pop.append(generate_tree(rng, d, method))

    return pop


def subtree_crossover(rng: random.Random, parent1: Node, parent2: Node, max_depth: int = 6) -> Tuple[Node, Node]:
    """
    Скрещивание поддеревьев (Subtree Crossover).
    Выбирает случайные точки разреза в обоих родителях и обменивает поддеревья,
    проверяя ограничение максимальной глубины.
    """
    child1 = parent1.clone()
    child2 = parent2.clone()

    nodes1 = child1.get_all_subtrees_with_parent()
    nodes2 = child2.get_all_subtrees_with_parent()

    # Предпочтение отдается функциональным узлам (90% по Koza), а не листьям
    funcs1 = [item for item in nodes1 if item[0].op_type == "function"]
    funcs2 = [item for item in nodes2 if item[0].op_type == "function"]

    cand1 = funcs1 if (funcs1 and rng.random() < 0.90) else nodes1
    cand2 = funcs2 if (funcs2 and rng.random() < 0.90) else nodes2

    node1, p1, idx1 = rng.choice(cand1)
    node2, p2, idx2 = rng.choice(cand2)

    # Производим замену
    if p1 is None:
        new_child1 = node2.clone()
    else:
        p1.children[idx1] = node2.clone()
        new_child1 = child1

    if p2 is None:
        new_child2 = node1.clone()
    else:
        p2.children[idx2] = node1.clone()
        new_child2 = child2

    # Проверка превышения максимальной глубины
    if new_child1.depth() > max_depth:
        new_child1 = parent1.clone()
    if new_child2.depth() > max_depth:
        new_child2 = parent2.clone()

    return new_child1, new_child2


def subtree_mutation(rng: random.Random, ind: Node, max_depth: int = 6, max_mut_depth: int = 2) -> Node:
    """Мутация поддерева: заменяет случайно выбранное поддерево на новое случайное поддерево."""
    res = ind.clone()
    nodes = res.get_all_subtrees_with_parent()
    node, parent, idx = rng.choice(nodes)
    
    new_subtree = generate_tree(rng, max_mut_depth, method="grow")
    if parent is None:
        res = new_subtree
    else:
        parent.children[idx] = new_subtree

    if res.depth() > max_depth:
        return ind.clone()
    return res


def point_mutation(rng: random.Random, ind: Node) -> Node:
    """Точечная мутация: заменяет оператор или терминал на однотипный того же ранга."""
    res = ind.clone()
    all_nodes = res.get_all_nodes()
    node = rng.choice(all_nodes)
    
    if node.op_type == "function":
        arity = len(node.children)
        same_arity = [f for f, a in FUNCTION_ARITY.items() if a == arity]
        node.value = rng.choice(same_arity)
    elif node.op_type == "variable":
        if rng.random() < 0.8:
            node.value = rng.choice(FEATURE_TERMINALS)
        else:
            node.op_type = "constant"
            node.value = rng.choice(CONSTANT_VALUES)
    elif node.op_type == "constant":
        if rng.random() < 0.5:
            node.value = rng.choice(CONSTANT_VALUES)
        else:
            node.op_type = "variable"
            node.value = rng.choice(FEATURE_TERMINALS)
    return res


def hoist_mutation(rng: random.Random, ind: Node) -> Node:
    """
    Hoist-мутация (анти-разрастание): выбирает случайное поддерево внутри особи
    и делает его новым корнем особи, тем самым резко сокращая размер дерева.
    """
    if ind.is_leaf():
        return ind.clone()
    all_nodes = ind.get_all_nodes()
    # Исключаем сам корень, чтобы мутация имела эффект
    subtrees = [n for n in all_nodes if n is not ind]
    if not subtrees:
        return ind.clone()
    chosen = rng.choice(subtrees)
    return chosen.clone()

