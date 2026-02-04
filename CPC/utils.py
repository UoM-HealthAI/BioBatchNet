import torch
import numpy as np
import random


def pairwise_loss(q1, q2, cons_type):
    """
    Compute pairwise constraint loss based on soft assignments.

    Args:
        q1, q2: soft assignments of paired samples
        cons_type: "ML" (must-link) or "CL" (cannot-link)
    """
    if cons_type == "ML":
        # Must-link: maximize similarity (dot product → 1)
        return torch.mean(-torch.log(torch.sum(q1 * q2, dim=1) + 1e-8))
    else:
        # Cannot-link: minimize similarity (dot product → 0)
        return torch.mean(-torch.log(1.0 - torch.sum(q1 * q2, dim=1) + 1e-8))


def generate_random_pair(y, num):
    """Generate random pairwise constraints."""
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []
    while num > 0:
        tmp1 = random.randint(0, y.shape[0] - 1)
        tmp2 = random.randint(0, y.shape[0] - 1)
        if tmp1 == tmp2:
            continue
        if y[tmp1] == y[tmp2]:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        else:
            cl_ind1.append(tmp1)
            cl_ind2.append(tmp2)
        num -= 1
    return np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)


def transitive_closure(ml_ind1, ml_ind2, cl_ind1, cl_ind2, n):
    """
    Calculate transitive closure for must-links and entailment for cannot-links.

    Args:
        ml_ind1, ml_ind2: instances within must-link constraint pairs
        cl_ind1, cl_ind2: instances within cannot-link constraint pairs
        n: total training instance number

    Returns:
        Transitive closure (must-links) and entailment (cannot-links)
    """
    ml_graph = {i: set() for i in range(n)}
    cl_graph = {i: set() for i in range(n)}

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in zip(ml_ind1, ml_ind2):
        add_both(ml_graph, i, j)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)

    for (i, j) in zip(cl_ind1, cl_ind2):
        add_both(cl_graph, i, j)
        for y in ml_graph[j]:
            add_both(cl_graph, i, y)
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)

    ml_res_set = set()
    cl_res_set = set()
    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                raise Exception('inconsistent constraints between %d and %d' % (i, j))
            if i <= j:
                ml_res_set.add((i, j))
            else:
                ml_res_set.add((j, i))
    for i in cl_graph:
        for j in cl_graph[i]:
            if i <= j:
                cl_res_set.add((i, j))
            else:
                cl_res_set.add((j, i))

    ml_res1, ml_res2 = [], []
    cl_res1, cl_res2 = [], []
    for (x, y) in ml_res_set:
        ml_res1.append(x)
        ml_res2.append(y)
    for (x, y) in cl_res_set:
        cl_res1.append(x)
        cl_res2.append(y)
    return np.array(ml_res1), np.array(ml_res2), np.array(cl_res1), np.array(cl_res2)
