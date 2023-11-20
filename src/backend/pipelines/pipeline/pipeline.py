from kedro.pipeline import Pipeline, pipeline
from .nodes import (
    drop_year_and_translation_node,
    drop_genres_node,
    drop_timestamp_node,
    index_ratings_node,
    normalize_ratings_node,
    train_model_node,
    recommend_movies_node,
)

actions = [
    drop_year_and_translation_node,
    drop_genres_node,
    drop_timestamp_node,
    index_ratings_node,
    normalize_ratings_node,
    train_model_node,
    recommend_movies_node,
]


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(actions)
