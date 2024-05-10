import cocoso
import numpy as np


def main():
    np.random.seed(48756)

    lambda_ = 0.5

    print(f'{",".join(cocoso.original_candidates)}')

    for _ in range(1000):
        weights = np.copy(cocoso.original_weights)
        np.random.shuffle(weights)
        print(
            f'{",".join(str(rank) for rank in cocoso.get_list(lambda_=lambda_, weights=weights))}'
        )


if __name__ == "__main__":
    main()
