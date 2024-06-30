from io import StringIO

import pandas

import numpy

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity


class Recommender():

    def __init__(
        self,
        data: str,
        categorical_features: list[str],
    ):
        self.data = data
        self.categorical_features = categorical_features    

    def _make_data_frame(self, data: str) -> pandas.Series | pandas.DataFrame:
        """
        Reads json data and returns it as a Pandas data frame
        """
        return pandas.read_json(StringIO(data))

    def _encode_features(
        self,
        data_frame: pandas.Series | pandas.DataFrame,
        features: list[str]
    ) -> numpy.ndarray:
        """
        Encodes categorical features with OneHotEncoder
        """
        return OneHotEncoder().fit_transform(data_frame[features]).toarray()

    def _get_similarity_matrix(self, features: numpy.ndarray) -> numpy.ndarray:
        """
        Returns cosine similarity of features provided
        """
        return cosine_similarity(features)

    def setup(self):
        """
        Must be run before any other operations
        """
        self.data_frame = self._make_data_frame(self.data)
        self.features = self._encode_features(self.data_frame, self.categorical_features)
        self.similarity_matrix = self._get_similarity_matrix(self.features)

    def recommend_for(self, index: int, limit: int) -> pandas.DataFrame:
        """
        Recommends a number of items from the dataset
        that are the most similar to the one provided
        """
        similarity_scores = list(enumerate(self.similarity_matrix[index]))
        
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        similar_items_indices = [i[0] for i in similarity_scores[1:limit+1]]

        return self.data_frame.iloc[similar_items_indices]  # type: ignore

    def get_index_by_feature(self, feature: str, value: str) -> int:
        """
        Returns index of matching row in the data frame based on unique feature.
        """
        matching_indeces: list[int] = self.data_frame.index[self.data_frame[feature] == value].tolist()
        
        if len(matching_indeces) == 0:
            raise ValueError("No matching name found")
        
        if len(matching_indeces) > 1:
            raise ValueError("Feature provided must be unique across rows")

        return matching_indeces[0]
