import numpy as np
import pandas as pd

from tqdm import tqdm


def interaction_matrix(interactions: pd.DataFrame, prob: bool = True) -> np.ndarray:
    """Return matrix , where X_ij is the probability that user i has listened to track j.

    :param prob: Use relative or absolute frequency.
    :param interactions: pd.DataFrame containing information about interactions between users and items.
    :return: 2D np.ndarray with 0 and 1 values.
    """
    res = np.zeros(shape=(max(interactions.user_id) + 1, max(interactions.track_id) + 1))
    for _, interaction in tqdm(interactions.iterrows(), desc="Constructing matrix", total=len(interactions)):
        res[interaction.user_id][interaction.track_id] = interaction.n_interactions
    if prob:
        for i, row in enumerate(res):
            res[i] /= (sum(row) or 1)
    return res
