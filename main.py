from load_data import load_all_data
from recommenders import *
from fire import Fire


def main(recommender: str = "top_k") -> None:
    targets, users, interactions, tracks = load_all_data()

    recommenders = {
        'top_k': TopKRecommender,
        'svd': SVDRecommender,
        'knn': KNNRecommender,
        'svd_ensemble': EnsembleSVD
    }

    print("Evaluating ", recommender)
    if recommender == 'svd_ensemble':
        recommender = EnsembleSVD(users, interactions, tracks, n_components=(10, 20, 40, 50, 90, 120, 150))
    else:
        recommender = recommenders[recommender](users, interactions, tracks)

    print(recommender.evaluate(n_splits=5, train_test_proportion=0.2))


if __name__ == '__main__':
    Fire(main)
