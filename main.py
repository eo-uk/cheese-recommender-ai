import json

from recommender import Recommender

from data import CHEESE_DATA


def main():
    recommender = Recommender(
        data=json.dumps(CHEESE_DATA),
        categorical_features=['country', 'milk', 'hardness', 'texture']
    )

    print("Initialising recommender..." "\n")

    recommender.setup()

    print("Recommender ready" "\n")

    while True:
        try:
            cheese_name = input("Enter cheese name:" "\n").title()
            cheese_index = recommender.get_index_by_feature("name", cheese_name)
            break
        except ValueError:
            print("Cheese not found" "\n")
    
    while True:
        try:
            limit = int(input("How many recommendations would you like to see?:" "\n"))
            break
        except ValueError:
            print("Invalid input, please enter digits only" "\n")

    print(f"======== Top {limit} Cheeses Similar to {cheese_name} ========")
    print(recommender.recommend_for(cheese_index, limit), "\n")

    input('Press Enter to exit')


if __name__ == "__main__":
    main()