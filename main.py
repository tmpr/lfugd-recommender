from load_data import load_all_data
from recommenders import *


def main() -> None:
    targets, users, interactions, tracks = load_all_data()

    knn_recommender = TopKRecommender(users, interactions, tracks)
    # top_k_recommender = TopKRecommender(users, interactions, tracks=tracks)
    # print("\nBaseline TopK:")
    # print(top_k_recommender.evaluate(n_splits=4))
    print(knn_recommender.evaluate(n_splits=4, train_test_proportion=0.3))


if __name__ == '__main__':
    main()
