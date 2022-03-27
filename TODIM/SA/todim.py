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
    theta=2.5,
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

    phi = np.zeros((n, m, m))

    weight_sum = sum(weights)

    for c in range(n):
        for i in range(m):
            for j in range(m):
                pic = raw_data[i, c]
                pjc = raw_data[j, c]
                val = 0.0
                if pic > pjc:
                    val = math.sqrt((pic - pjc) * weights[c] / weight_sum)
                if pic < pjc:
                    val = (
                        -1.0 / theta * math.sqrt(weight_sum * (pjc - pic) / weights[c])
                    )
                phi[c, i, j] = val

    delta = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            delta[i, j] = sum(phi[:, i, j])

    delta_sums = np.zeros(m)
    for i in range(m):
        delta_sums[i] = sum(delta[i, :])

    delta_min = min(delta_sums)
    delta_max = max(delta_sums)

    ratings = (delta_sums - delta_min) / (delta_max - delta_min)
    return rank_according_to(ratings, candidates)


def rank_according_to(data, candidates):
    # return (len(candidates) + 1 - rankdata(data)).astype(int)
    ranks = (rankdata(data) - 1).astype(int)
    storage = np.zeros_like(candidates)
    storage[ranks] = candidates
    return storage[::-1]
