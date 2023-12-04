from kedro.pipeline import Pipeline, pipeline
from .nodes import (
    drop_genres_and_title_node,
    drop_timestamp_node,
    index_ratings_node,
    create_tmdb_mapping_node,
    normalize_ratings_node,
    train_model_node,
    recommend_movies_node,
    drop_imdb_column_node,
)

actions = [
    drop_genres_and_title_node,
    drop_timestamp_node,
    drop_imdb_column_node,
    index_ratings_node,
    create_tmdb_mapping_node,
    normalize_ratings_node,
    train_model_node,
    recommend_movies_node,
]


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(actions)
