from load_data import load_all_data
from recommenders import SVDRecommender, TopKRecommender


def main() -> None:
    targets, users, interactions, tracks = load_all_data()

    svd_recommender = SVDRecommender(users, interactions, tracks)
    top_k_recommender = TopKRecommender(users, interactions, tracks=tracks)
    print("\nBaseline TopK:")
    print(top_k_recommender.evaluate(n_splits=2))
    print(svd_recommender.evaluate(n_splits=4))


if __name__ == '__main__':
    main()
