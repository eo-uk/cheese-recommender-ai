from io import StringIO

import pandas

import numpy

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


class Recommender():

    def __init__(
        self,
        data: str,
        categorical_features: list[str] | None = None,
        numerical_features: list[int] | None = None,
    ):
        self.data: str = data
        self.categorical_features: list[str] | None = categorical_features
        self.numerical_features: list[int] | None = numerical_features    

    def _make_data_frame(self, data: str) -> pandas.Series | pandas.DataFrame:
        """
        Reads json data and returns it as a Pandas data frame
        """
        return pandas.read_json(StringIO(data))

    def _encode_categorical_features(
        self,
        data_frame: pandas.Series | pandas.DataFrame,
        categorical_features: list[str] | None
    ) -> pandas.DataFrame:
        """
        Encodes categorical features with OneHotEncoder
        """
        if not categorical_features:
            return pandas.DataFrame()

        encoded_array = OneHotEncoder().fit_transform(data_frame[categorical_features]).toarray()
        return pandas.DataFrame(encoded_array)

    def _scale_numerical_features(
        self,
        data_frame,
        numerical_features,
    ) -> pandas.DataFrame:
        """
        Scales numerical features with Standard Scaler
        """
        if not numerical_features:
            return pandas.DataFrame()

        scaled_numerical_features = StandardScaler().fit_transform(data_frame[numerical_features])

        return pandas.DataFrame(scaled_numerical_features, columns=numerical_features)

    def _get_similarity_matrix(self, features: pandas.DataFrame) -> numpy.ndarray:
        """
        Returns cosine similarity of features provided
        """
        return cosine_similarity(features)

    def _get_features(
        self,
        data_frame: pandas.Series | pandas.DataFrame,
        categorical_features: list[str] | None,
        numerical_features: list[int]  | None,
    ) -> pandas.DataFrame:
        """
        Processes categorical and numerical features and returns them as a single data frame 
        """
        encoded_features = self._encode_categorical_features(data_frame, categorical_features)
        scaled_features = self._scale_numerical_features(data_frame, numerical_features)
        return self._concat_features(encoded_features, scaled_features)

    def _concat_features(
        self,
        categorical_features: pandas.DataFrame,
        numerical_features: pandas.DataFrame
    ) -> pandas.DataFrame:
        """
        Concatenates features into a single data frame
        """
        return pandas.concat([numerical_features, categorical_features], axis=1)

    def setup(self):
        """
        Must be run before any other operations
        """
        self.data_frame = self._make_data_frame(self.data)
        self.features = self._get_features(
            self.data_frame,
            self.categorical_features,
            self.numerical_features
        )
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
