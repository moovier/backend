from kedro.pipeline import Pipeline, pipeline
from .nodes import (
    drop_genres_and_title_node,
    drop_timestamp_node,
    index_ratings_node,
    create_tmdb_mapping_node,
    normalize_ratings_node,
    build_model_node,
    recommend_movies_node,
    drop_imdb_column_node,
    train_model_node,
    optuna_model_node,
    optuna_recommend_movies_node,
    train_model_node,
    pycaret_merge_datasets_node,
    pycaret_predict_ratings_node,
)

actions = [
    drop_genres_and_title_node,
    drop_timestamp_node,
    drop_imdb_column_node,
    index_ratings_node,
    create_tmdb_mapping_node,
    normalize_ratings_node,
    build_model_node,
    train_model_node,
    recommend_movies_node,
    optuna_model_node,
    optuna_recommend_movies_node,
    pycaret_merge_datasets_node,
    pycaret_predict_ratings_node,
]


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(actions)
