import numpy as np
import todim


def main():
    np.random.seed(1249871)
    thetas = np.linspace(1.0, 1001.0, endpoint=True, num=1000)
    print(f'Theta,{",".join(todim.candidates)}')
    for theta in thetas:
        print(
            f'{theta:.2f},{",".join(str(rank) for rank in todim.get_list(theta, np.copy(todim.original_weights)))}'
        )


if __name__ == "__main__":
    main()
