import cpt_todim
import numpy as np


def main():
    np.random.seed(48756)

    alpha = 0.88
    beta = 0.88
    theta = 0.61
    mu = 0.69
    sigma = 2.25

    print(f'{",".join(cpt_todim.candidates)}')

    for _ in range(1000):
        weights = np.copy(cpt_todim.original_weights)
        np.random.shuffle(weights)
        print(
            f'{",".join(str(rank) for rank in cpt_todim.get_list(alpha,beta,theta, mu,sigma, weights))}'
        )


if __name__ == "__main__":
    main()
