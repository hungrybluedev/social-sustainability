import math  # For sqrt and other stuff
import numpy as np  # For linear algebra
import pandas as pd  # For tabular output
from scipy.stats import rankdata  # For ranking the candidates based on score

attributes_data = pd.read_csv("../../data/criteria.csv")


def _initialize_items():
    benefit_attributes = set()
    attributes = []
    rankings = []
    n = 0

    for i, row in attributes_data.iterrows():
        attributes.append(row["Indicator"])
        rankings.append(row["Rank"])
        n += 1

        if row["Ideally"] == "Higher":
            benefit_attributes.add(i)

    rankings = np.array(rankings)
    weights = 2 * (n + 1 - rankings) / (n * (n + 1))
    return benefit_attributes, attributes, weights


benefit_attributes, attributes, original_weights = _initialize_items()

original_dataframe = pd.read_csv("../../data/alternatives.csv").T

updated_dataframe = original_dataframe.drop(original_dataframe.index[0])

original_candidates = np.array(updated_dataframe.index)


def get_list(
    alpha=0.88,
    beta=0.88,
    gamma=0.61,
    delta=0.69,
    lambda_=2.25,
    weights=original_weights,
    candidates=original_candidates,
    raw_data=updated_dataframe.to_numpy(),
):
    [m, n] = raw_data.shape

    for j in range(n):
        column = raw_data[:, j]
        if j in benefit_attributes:
            raw_data[:, j] /= sum(column)
        else:
            column = 1 / column
            raw_data[:, j] = column / sum(column)

    max_weight = max(weights)
    weights /= max_weight

    inv_gamma = 1 / gamma
    inv_delta = 1 / delta

    pi = np.zeros((m, m, n))

    for i in range(m):
        for k in range(m):
            for j in range(n):
                if raw_data[i, j] >= raw_data[k, j]:
                    w_gamma = weights[j] ** gamma
                    pi[i, k, j] = w_gamma / (
                        (w_gamma + (1 - weights[j]) ** gamma) ** inv_gamma
                    )
                else:
                    w_delta = weights[j] ** delta
                    pi[i, k, j] = w_delta / (
                        (w_delta + (1 - weights[j]) ** delta) ** inv_delta
                    )

            pi[i, k, :] /= max(pi[i, k, :])

    pi_sums = np.zeros((m, m))

    for i in range(m):
        for k in range(m):
            pi_sums[i, k] = sum(pi[i, k, :])

    phi = np.zeros((n, m, m))

    for i in range(m):
        for k in range(m):
            for j in range(n):
                x_ij = raw_data[i, j]
                x_kj = raw_data[k, j]
                val = 0.0
                if x_ij > x_kj:
                    val = pi[i, k, j] * ((x_ij - x_kj) ** alpha) / pi_sums[i, k]
                if x_ij < x_kj:
                    val = (
                        -lambda_ * pi_sums[i, k] * ((x_kj - x_ij) ** beta) / pi[i, k, j]
                    )
                phi[j, i, k] = val

    big_phi = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            big_phi[i, j] = sum(phi[:, i, j])

    big_phi_sums = np.zeros(m)
    for i in range(m):
        big_phi_sums[i] = sum(big_phi[i, :])

    big_phi_min = min(big_phi_sums)
    big_phi_max = max(big_phi_sums)

    ratings = (big_phi_sums - big_phi_min) / (big_phi_max - big_phi_min)
    return rank_according_to(ratings, candidates)


def rank_according_to(data, candidates):
    return (len(candidates) + 1 - rankdata(data)).astype(int)


def rank_according_to(data, candidates):
    # return (len(candidates) + 1 - rankdata(data)).astype(int)
    ranks = (rankdata(data) - 1).astype(int)
    storage = np.zeros_like(candidates)
    storage[ranks] = candidates
    return storage[::-1]
