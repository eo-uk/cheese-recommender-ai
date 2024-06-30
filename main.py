import json

from recommender import Recommender

from data.cheese import DATA as CHEESE_DATA
from data.movie import DATA as MOVIES_DATA


def main():

    while True:
        try:
            dataset_name = input("What dataset would you like to use? (Cheese, Movie):" "\n").lower()
            if dataset_name in ["cheese", "movie"]:
                break
        except ValueError:
            print("Incorrect dataset name" "\n")

    if dataset_name == "cheese":
        dataset = CHEESE_DATA
        categorical_features = ['country', 'milk', 'hardness', 'texture']
        numerical_features= None
        unique_feature = "name"
    elif dataset_name == "movie":
        dataset = MOVIES_DATA
        categorical_features=['title', 'genre', 'director']
        numerical_features=['year']
        unique_feature = "title"

    recommender = Recommender(
        data=json.dumps(dataset),
        categorical_features=categorical_features,
        numerical_features=numerical_features,
    )

    print("Initialising recommender..." "\n")
    recommender.setup()
    print("Recommender ready" "\n")

    while True:
        try:
            item_name = input(f"Enter {dataset_name} {unique_feature}" "\n").title()
            item_index = recommender.get_index_by_feature(unique_feature, item_name)
            break
        except ValueError:
            print("Item not found" "\n")
    
    while True:
        try:
            limit = int(input("How many recommendations would you like to see?:" "\n"))
            break
        except ValueError:
            print("Invalid input, please enter digits only" "\n")

    print(f"======== Top {limit} {dataset_name.title()+'s'} Similar to {item_name} ========")
    print(recommender.recommend_for(item_index, limit), "\n")

    input('Press Enter to exit')


if __name__ == "__main__":
    main()