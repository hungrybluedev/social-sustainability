import cpt_todim
import numpy as np


def main():
    old_candidates = cpt_todim.original_candidates.copy()
    old_ranks = cpt_todim.get_list()

    for candidate in old_candidates:
        removed_ranks = np.delete(old_ranks, np.where(old_ranks == candidate))

        new_candidates = np.delete(
            old_candidates, np.where(old_candidates == candidate)
        )

        new_dataframe = cpt_todim.original_dataframe.loc[new_candidates].to_numpy()

        new_ranks = cpt_todim.get_list(
            candidates=new_candidates, raw_data=new_dataframe
        )
        if not np.array_equal(removed_ranks, new_ranks):
            print(f"Removed {candidate}: {removed_ranks} vs {new_ranks}")


if __name__ == "__main__":
    main()
