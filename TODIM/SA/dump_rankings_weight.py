import todim
import numpy as np


def main():
    np.random.seed(48756)
    theta = 2.5
    print(f'{",".join(todim.candidates)}')

    for _ in range(1000):
        weights = np.copy(todim.original_weights)
        np.random.shuffle(weights)
        print(f'{",".join(str(rank) for rank in todim.get_list(theta, weights))}')


if __name__ == "__main__":
    main()
