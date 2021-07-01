import numpy as np
import pandas as pd

from tqdm import tqdm


def interaction_matrix(users: pd.DataFrame, items: pd.DataFrame, interactions: pd.DataFrame) -> np.ndarray:
    """Return matrix , where X_ij is the probability that user i has listened to track j.

    :param users: pd.DataFrame containing user information.
    :param items: pd.DataFrame containing track information.
    :param interactions: pd.DataFrame containing information about interactions between users and items.
    :return: 2D np.ndarray with 0 and 1 values.
    """
    res = np.zeros(shape=(len(users) + 100, len(items) + 100))
    for _, interaction in tqdm(interactions.iterrows(), desc="Constructing matrix", total=len(interactions)):
        res[interaction.user_id][interaction.track_id] = interaction.n_interactions
    for i, row in enumerate(res):
        res[i] /= (sum(row) or 1)
    return res
