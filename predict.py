from load_data import load_all_data
from recommenders import *


def main() -> None:
    targets, users, interactions, tracks = load_all_data()
    recommender = EnsembleSVD(users, interactions, tracks, n_components=(10, 20, 40, 50, 90, 120, 150))
    recommendation_file = Path('rec_Alexander-Temper_11905006.tsv')
    for target in tqdm(targets.user_id, total=len(targets), desc="Recommending ..."):
        rec = recommender.recommend(user_id=target, k=15)
        rec.append_to(recommendation_file)


if __name__ == '__main__':
    main()
