from pathlib import Path
from typing import Tuple
from tqdm import tqdm

import pandas as pd

DATA_FOLDER = Path('data')
TARGETS_PATH = DATA_FOLDER / 'target_users.tsv'
USER_INFO_PATH = DATA_FOLDER / 'user_info.tsv'
INTERACTIONS_PATH = DATA_FOLDER / 'interactions.tsv'
TRACK_INFO_PATH = DATA_FOLDER / 'track_info.tsv'


def load_all_data() -> Tuple[pd.DataFrame, ...]:
    """Return `targets`, `users`, `interactions`, `tracks`."""
    return tuple(
        pd.read_csv(path, sep='\t') for path in (TARGETS_PATH, USER_INFO_PATH, INTERACTIONS_PATH, TRACK_INFO_PATH))


def split(interactions: pd.DataFrame, p: float = 0.25) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return train and test interactions.

    Args:
        interactions (pd.DataFrame): Interaction Dataframe.
        p (float, optional): Percentage of samples going into the test-set. Defaults to 0.25.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test set.
    """
    test = interactions.groupby('track_id').sample(frac=p)
    rows = set((a, b) for _, (a, b, _) in test.iterrows())
    train_mask = [i for i, (_, (a, b, _)) in tqdm(enumerate(interactions.iterrows()), desc="Constructing train-set",
                                                  total=len(interactions)) if (a, b) not in rows]
    train = interactions.iloc[train_mask]

    return train, test
