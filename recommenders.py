from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import sklearn.decomposition as decomp
import sklearn.manifold as mf
from tqdm import tqdm

from load_data import split
from utils import interaction_matrix


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
    computed: List[int] = None

    def __str__(self) -> str:
        return str(self.user_id) + '\t' + ','.join(str(track_id) for track_id in self.track_ids)

    def append_to(self, path: Path) -> None:
        with path.open('a') as p:
            p.write(str(self) + '\n')


class Recommender:

    def __init__(self, users: pd.DataFrame, interactions: pd.DataFrame, tracks: pd.DataFrame,
                 matrix: np.ndarray = None) -> None:
        self.users = users
        self.interactions = interactions
        self.tracks = tracks
        self.full_matrix = matrix if matrix is not None else interaction_matrix(self.interactions)

    def recommend(self, user_id: int, k: int, matrix: np.array = None) -> Recommendation:
        return Recommendation(user_id=user_id, track_ids=[-1])

    def prepare_eval(self, train_matrix: np.ndarray = None) -> None:
        pass

    def end_eval(self):
        pass

    def evaluate(self, n_splits: int = 5, train_test_proportion: float = 0.2) -> np.ndarray:
        mrr_s = []
        interactions = self.interactions
        users = self.users.iloc[:max(interactions.user_id)]

        for _ in range(n_splits):
            train, test = split(interactions, p=train_test_proportion)

            test_matrix = interaction_matrix(interactions=test)
            train_matrix = interaction_matrix(interactions=train)

            self.prepare_eval(train_matrix)

            mrr_s.append(self.mrr(train_matrix, test_matrix))
            print(mrr_s[-1])

        self.end_eval()

        return np.mean(mrr_s)

    def mrr(self, train_matrix, test_matrix) -> float:
        mrr = 0

        for i, user_interactions in tqdm(enumerate(test_matrix), desc=f"Computing MRR split", total=len(test_matrix)):
            recommendations = self.recommend(i, 15, matrix=train_matrix)
            rank_result = self.rank(user_interactions, recommendations.track_ids)
            mrr += 1 / rank_result

        return mrr / len(test_matrix)

    @staticmethod
    def rank(user_interactions: pd.Series, recommendations: Iterable[int]) -> Real:
        for i, track_id in enumerate(recommendations, start=1):
            if user_interactions[track_id]:
                return i
        return float('inf')


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


class KNNRecommender(Recommender):

    def __init__(self, users: pd.DataFrame, interactions: pd.DataFrame, tracks: pd.DataFrame,
                 k_neighbors: int = 10) -> None:
        super().__init__(users, interactions, tracks)
        self.k_neighbors = k_neighbors

    def nearest_neighbors(self, user_id: int, matrix: np.ndarray = None) -> np.ndarray:
        user = matrix[user_id]
        closest = sorted(range(len(matrix)), key=lambda other_user_id:
        np.linalg.norm(user - matrix[other_user_id], ord=2))[:self.k_neighbors]

        return np.array([neighbor for neighbor in closest if user_id != neighbor])

    def recommend(self, user_id: int, k: int, matrix: np.array = None) -> Recommendation:
        matrix = matrix if matrix is not None else self.full_matrix
        neighbors = self.nearest_neighbors(user_id, matrix)
        centroid = np.mean(matrix[neighbors], axis=0)

        user_has_seen = matrix[user_id]

        if not any(track for track in user_has_seen):
            track_ids = [track_id for track_id in sorted_by_popularity(matrix) if not user_has_seen[track_id]][:k]
        else:
            track_ids = sorted(range(len(centroid)), key=lambda i: centroid[i])
            track_ids = [track_id for track_id in track_ids if not user_has_seen[track_id]]

        return Recommendation(user_id, track_ids[:k])


class SVDRecommender(Recommender):

    def __init__(self, users: pd.DataFrame, interactions: pd.DataFrame, tracks: pd.DataFrame, n_components: int = 100,
                 matrix: np.ndarray = None) -> None:
        super(SVDRecommender, self).__init__(users, interactions, tracks, matrix)
        self.n_components = n_components
        self.compute_svd(matrix=self.full_matrix)

    def compute_svd(self, matrix: np.ndarray):
        svd = decomp.TruncatedSVD(n_components=self.n_components)
        svd.fit(matrix)
        self.U = svd.transform(matrix)
        self.V = svd.components_
        var_ratios = svd.explained_variance_ratio_

        for i, cum_ratio in enumerate(np.cumsum(var_ratios)):
            k = i
            if cum_ratio >= 0.9:
                break

        self.sigma = np.diag(svd.singular_values_[:k])

        self.U = self.U[:, :k]
        self.V = self.V[:k, :]

    def prepare_eval(self, train_matrix: np.ndarray = None) -> None:
        self.compute_svd(matrix=train_matrix)

    def end_eval(self):
        self.compute_svd(matrix=self.full_matrix)

    def recommend(self, user_id: int, k: int, matrix: np.array = None) -> Recommendation:
        approximated_ratings = self.infer(user_id)
        ids = range(len(approximated_ratings))

        track_ids = sorted(ids, key=lambda i: approximated_ratings[i], reverse=True)
        track_ids = [id_ for id_ in track_ids if not matrix[user_id][id_]][:k]

        return Recommendation(user_id, track_ids, computed=approximated_ratings)

    def infer(self, user_id: int):
        return self.U[user_id] @ self.sigma @ self.V


class EnsembleSVD(Recommender):

    def __init__(self, users: pd.DataFrame, interactions: pd.DataFrame,
                 tracks: pd.DataFrame, n_components: Tuple[int, ...] = (10, 50, 100, 150, 200)) -> None:
        super().__init__(users, interactions, tracks)
        self.recommenders = [SVDRecommender(users, interactions, tracks, n_components=n, matrix=self.full_matrix)
                             for n in tqdm(n_components, total=len(n_components), desc="Computing sub-svd.")]

    def prepare_eval(self, train_matrix: np.ndarray = None) -> None:
        for rec in self.recommenders:
            rec.prepare_eval(train_matrix)

    def end_eval(self):
        for rec in self.recommenders:
            rec.end_eval()

    def recommend(self, user_id: int, k: int, matrix: np.array = None) -> Recommendation:
        matrix = matrix if matrix is not None else self.full_matrix

        recommendations = [rec.infer(user_id) for rec in self.recommenders]
        recommendations = np.mean(recommendations, axis=0)

        ids = range(len(recommendations))

        track_ids = sorted(ids, key=lambda i: recommendations[i], reverse=True)
        track_ids = [id_ for id_ in track_ids if not matrix[user_id][id_]][:k]

        return Recommendation(user_id, track_ids)


class TSNERecommender(KNNRecommender):

    def __init__(self, users: pd.DataFrame, interactions: pd.DataFrame, tracks: pd.DataFrame, k_neighbors: int) -> None:
        super().__init__(users, interactions, tracks, k_neighbors=k_neighbors)
        self.downprojected = None

    def prepare_eval(self, train_matrix: np.ndarray = None) -> None:
        self.downprojected = mf.TSNE(n_components=3, perplexity=50).fit_transform(train_matrix)

    def nearest_neighbors(self, user_id: int, matrix: np.ndarray = None) -> np.ndarray:
        return super(TSNERecommender, self).nearest_neighbors(user_id, matrix=self.downprojected)
