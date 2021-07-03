from abc import abstractmethod
from dataclasses import dataclass
from utils import interaction_matrix
from load_data import split
from numbers import Real
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from tqdm import tqdm


def sorted_by_popularity(matrix) -> list:
    """Return `track_id`s sorted by popularity.
    :rtype: List containing integers.
    """
    num_interactions = matrix.sum(axis=0)
    return sorted(range(len(num_interactions)), key=lambda i: num_interactions[i], reverse=True)


def most_popular(matrix: np.ndarray, n=10) -> list:
    """Return track_id of `n` most popular (most broadly interacted with) songs."""
    return sorted_by_popularity(matrix)[:n]


@dataclass
class Recommendation:
    user_id: int
    track_ids: List[int]

    def __str__(self) -> str:
        return str(self.user_id) + '\t' + ','.join(str(track_id) for track_id in self.track_ids)

    def append_to(self, path: Path) -> None:
        with path.open('a') as p:
            p.write(str(self))


def rank(user_interactions: pd.Series, recommendations: Iterable[int]) -> Real:
    for i, track_id in enumerate(recommendations, start=1):
        if user_interactions[track_id]:
            return i
    return float('inf')


class Recommender:

    def __init__(self, users: pd.DataFrame, interactions: pd.DataFrame, tracks: pd.DataFrame) -> None:
        self.users = users
        self.interactions = interactions
        self.tracks = tracks
        self.full_matrix = interaction_matrix(self.users, self.tracks, self.interactions)

    @abstractmethod
    def recommend(self, user_id: int, k: int, matrix: np.array = None) -> Recommendation:
        pass

    def evaluate(self, n_splits: int = 5, train_test_proportion: float = 0.2) -> np.ndarray:
        mrr_s = []
        interactions = self.interactions.iloc[:5_000]
        users = self.users.iloc[:max(interactions.user_id)]

        for _ in range(n_splits):
            train, test = split(interactions, p=train_test_proportion)

            test_matrix = interaction_matrix(users, items=self.tracks, interactions=test)
            train_matrix = interaction_matrix(users, items=self.tracks, interactions=train)

            mrr_s.append(self.mrr(train_matrix, test_matrix))

        return np.mean(mrr_s)

    def mrr(self, train_matrix, test_matrix) -> float:
        mrr = 0

        for i, user_interactions in tqdm(enumerate(test_matrix), desc=f"Computing MRR split", total=len(test_matrix)):
            recommendations = self.recommend(i, 15, matrix=train_matrix)
            rank_result = rank(user_interactions, recommendations.track_ids)
            mrr += 1 / rank_result

        return mrr / len(test_matrix)


class TopKRecommender(Recommender):

    def recommend(self, user_id: int, k: int, matrix: np.ndarray = None) -> Recommendation:
        """Given user, recommend top `k` most popular tracks which the user has not yet listened to.

        Args:
            user_id (int): User to recommend to.
            k (int): Number of items to recommend.
            matrix (np.ndarray): Recommendations to "train on".

        Returns:
            Recommendation: Dataclass containing recommendations and `user_id`.
        """
        matrix = matrix if matrix is not None else self.full_matrix
        user_has_seen = matrix[user_id]
        return Recommendation(user_id,
                              [track_id for track_id in sorted_by_popularity(matrix)
                               if not user_has_seen[track_id]][:k])


class SVDRecommender(Recommender):

    def recommend(self, user_id: int, k: int, matrix: np.array = None) -> Recommendation:
        return super().recommend(user_id, k, matrix=matrix)
