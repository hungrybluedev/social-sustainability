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
    lambda_=0.5,
    weights=original_weights,
    candidates=original_candidates,
    raw_data=updated_dataframe.to_numpy(),
):
    [m, n] = raw_data.shape

    max_vals = np.amax(raw_data, axis=0)
    min_vals = np.amin(raw_data, axis=0)

    for j in range(n):
        column = raw_data[:, j]
        denominator = max_vals[j] - min_vals[j]

        if j in benefit_attributes:
            raw_data[:, j] = (raw_data[:, j] - min_vals[j]) / denominator
        else:
            raw_data[:, j] = (max_vals[j] - raw_data[:, j]) / denominator

    sum_values_full = np.zeros((m, n))
    pow_values_full = np.zeros((m, n))

    for i in range(m):
        sum_values_full[i, :] = weights * raw_data[i, :]
        pow_values_full[i, :] = weights ** raw_data[i, :]

    sum_values = np.sum(sum_values_full, axis=1)
    pow_values = np.sum(pow_values_full, axis=1)

    ma_denom = sum(sum_values) + sum(pow_values)

    m_a = (sum_values + pow_values) / ma_denom

    min_sum = min(sum_values)
    min_pow = min(pow_values)

    m_b = sum_values / min_sum + pow_values / min_pow

    lambda_ = 0.5

    max_sum = max(sum_values)
    max_pow = max(pow_values)

    one_minus_lambda = 1 - lambda_

    mc_denom = lambda_ * max_sum + one_minus_lambda * max_pow

    m_c = (lambda_ * sum_values + one_minus_lambda * pow_values) / mc_denom

    one_third = 1.0 / 3.0
    m_vals = (m_a * m_b * m_c) ** one_third + one_third * (m_a + m_b + m_c)

    return rank_according_to(m_vals, candidates)


def rank_according_to(data, candidates):
    # return rankdata(data).astype(int)
    ranks = (rankdata(data) - 1).astype(int)
    storage = np.zeros_like(candidates)
    storage[ranks] = candidates
    return storage
