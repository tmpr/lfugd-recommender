from typing import List, Tuple
import pandas as pd
from pathlib import Path

from pandas.core.algorithms import SelectNFrame


DATA_FOLDER = Path('data')
TARGETS_PATH = DATA_FOLDER / 'target_users.tsv'
USER_INFO_PATH = DATA_FOLDER / 'user_info.tsv'
INTERACTIONS_PATH = DATA_FOLDER / 'interactions.tsv'
TRACK_INFO_PATH = DATA_FOLDER / 'track_info.tsv'

def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return `targets`, `user_info`, `interactions`, `track_info`."""
    return (pd.read_csv(path, sep='\t') for path in (TARGETS_PATH, USER_INFO_PATH, INTERACTIONS_PATH, TRACK_INFO_PATH))


def has_seen(user_id: int, track_id: int, interactions: pd.DataFrame) -> bool:
    raise NotImplementedError()

class Recommendation:

    def __init__(self, user_id: int, recommendations: List[int], interactions: pd.DataFrame) -> None:
        if len(recommendations) > 15:
            raise ValueError("Invalid number of recommendations")
        elif any(has_seen(user_id, track_id, interactions) for track_id in recommendations):
            raise ValueError(f"User {user_id} has invalid recommendations: {recommendations}.")
        self.user_id = user_id
        self.recommendations = recommendations

    def __str__(self) -> str:
        return str(self.user_id) + '\t' + ','.join(str(track_id) for track_id in self.recommendations)

    def append_to(self, path: Path) -> None:
        with path.open('a') as p:
            p.write(str(self))


class Recommender:
    pass


