from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


def _holm(pvals):
    """Correção de Holm-Bonferroni para comparações múltiplas."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def statistical_test(data, alg_names, reference=None, pv_friedman=0.05, pv_wilcoxon=0.05,
                     correction="holm", verbose=True):
    """
    Friedman seguido de Wilcoxon pareado (se o Friedman for significativo).

    data: matriz `algoritmos x amostras`; as amostras precisam estar pareadas
          (mesma ordem de folds/métricas em todas as linhas).
    reference: nome do algoritmo de referência. Se informado, compara só reference vs. cada
               outro (o que vai na tabela do artigo); se None, compara todos os pares.
    correction: "holm" ou None.

    Retorna (friedman_p, DataFrame com uma linha por comparação).
    """
    lengths = {len(r) for r in data}
    if len(lengths) != 1:
        raise ValueError(f"Todas as linhas precisam ter o mesmo número de amostras; obtido: {lengths}")
    data = np.asarray(data, float)
    if data.shape[0] != len(alg_names):
        raise ValueError("data e alg_names devem ter o mesmo número de linhas")

    _, p_friedman = friedmanchisquare(*data)
    if verbose:
        print(f"Friedman: p = {p_friedman:.3g} ({data.shape[0]} algoritmos, {data.shape[1]} amostras)")

    if reference is not None:
        r = alg_names.index(reference)
        pairs = [(r, j) for j in range(len(alg_names)) if j != r]
    else:
        pairs = list(combinations(range(len(alg_names)), 2))

    rows = []
    for i, j in pairs:
        diff = data[i] - data[j]
        _, p = wilcoxon(data[i], data[j])
        rows.append({
            "alg_a": alg_names[i], "alg_b": alg_names[j],
            "median_diff_a_minus_b": float(np.median(diff)),
            "wins_a": int((diff > 0).sum()), "wins_b": int((diff < 0).sum()),
            "p_value": float(p),
        })
    res = pd.DataFrame(rows)
    res["p_adjusted"] = _holm(res["p_value"]) if correction == "holm" else res["p_value"]
    run_pairwise = p_friedman <= pv_friedman
    res["significant"] = run_pairwise & (res["p_adjusted"] < pv_wilcoxon)
    res["better"] = np.where(~res["significant"], "-",
                             np.where(res["median_diff_a_minus_b"] > 0, res["alg_a"], res["alg_b"]))
    if not run_pairwise and verbose:
        print("Friedman não significativo: comparações pareadas não são interpretadas.")
    if verbose:
        print(res.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    return p_friedman, res