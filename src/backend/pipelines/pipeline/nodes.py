import numpy as np
import pandas as pd
from kedro.pipeline import node

from .model import train_recommender


def drop_genres_and_title(movies):
    movies = movies.drop(columns="genres")
    movies = movies.drop(columns="title")
    return movies


def drop_imbd_column(links):
    return links.drop(columns="imdbId")


def drop_timestamp(ratings):
    return ratings.drop(columns="timestamp")


def index_ratings(ratings):
    def create_index_maps(data_column):
        unique_values = data_column.unique().tolist()

        index_to_value_map = dict(enumerate(unique_values))
        value_to_index_map = dict(map(reversed, index_to_value_map.items()))

        return index_to_value_map, value_to_index_map

    _, users_to_indices = create_index_maps(ratings.userId)
    indices_to_movies, movies_to_indices = create_index_maps(ratings.movieId)

    ratings["user"] = ratings.userId.map(users_to_indices)
    ratings["movie"] = ratings.movieId.map(movies_to_indices)
    ratings["rating"] = ratings.rating.values.astype(np.float32)

    return ratings, movies_to_indices, indices_to_movies, users_to_indices


def create_tmdb_mapping(links):
    return dict(zip(links["movieId"], links["tmdbId"]))


def normalize_ratings(ratings):
    min_rating, max_rating = min(ratings.rating), max(ratings.rating)

    def normalizer(x):
        return (x - min_rating) / (max_rating - min_rating)

    ratings["rating"] = ratings["rating"].apply(normalizer).values
    ratings.sample(frac=1, random_state=42)
    return ratings


def train_model(ratings, training_split_ratio, embedding_size, learning_rate, patience):
    return train_recommender(
        x=ratings[["user", "movie"]].values,
        y=ratings["rating"],
        num_users=ratings["user"].nunique(),
        num_movies=ratings["movie"].nunique(),
        embedding_size=embedding_size,
        learning_rate=learning_rate,
        train_split=int(training_split_ratio * ratings.shape[0]),
        patience=patience,
    )


def recommend_movies(
    model,
    movies,
    movies_to_tmdb,
    ratings,
    movies_to_indices,
    indices_to_movies,
    users_to_indices,
    top_k,
    user_ids
):
    recommended_movies_for_all_users = []
    for user_id in user_ids:
        movies_watched = ratings[ratings["userId"] == user_id]

        movies_not_watched = ~movies["movieId"].isin(movies_watched["movieId"].values)
        movies_not_watched = movies[movies_not_watched]["movieId"]
        movies_not_watched = set(movies_not_watched) & set(movies_to_indices.keys())
        movies_not_watched = list(movies_not_watched)
        movies_not_watched = [
            [movies_to_indices.get(movie)] for movie in movies_not_watched
        ]

        user_encoder = users_to_indices.get(user_id)

        inputs = ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
        inputs = np.hstack(inputs)

        ratings_pred = model.predict(inputs)
        metrics = model.evaluate(x=inputs, y=ratings_pred, return_dict=True)

        ratings_pred = ratings_pred.flatten()
        ratings_sorted_indices = ratings_pred.argsort()[-top_k:][::-1]

        recommended_movies = [
            indices_to_movies.get(movies_not_watched[rating][0])
            for rating in ratings_sorted_indices
        ]

        recommended_movies = movies[movies["movieId"].isin(recommended_movies)]["movieId"]
        recommended_movies = [movies_to_tmdb.get(rec) for rec in recommended_movies]
        recommended_movies = [str(int(rec)) for rec in recommended_movies]

        column = {
            "user": user_id,
            "recommendations": ",".join(recommended_movies),
            "binary_crossentropy": metrics.get('binary_crossentropy'),
            "loss": metrics.get('loss')
        }

        recommended_movies_for_all_users.append(column)
    return pd.DataFrame(recommended_movies_for_all_users)


drop_genres_and_title_node = node(
    func=drop_genres_and_title,
    inputs="movies",
    outputs="cleaned_movies",
    name=drop_genres_and_title.__name__,
)

drop_imdb_column_node = node(
    func=drop_imbd_column,
    inputs="links",
    outputs="cleaned_links",
    name=drop_imbd_column.__name__,
)

drop_timestamp_node = node(
    func=drop_timestamp,
    inputs="ratings",
    outputs="cleaned_ratings",
    name=drop_timestamp.__name__,
)

create_tmdb_mapping_node = node(
    func=create_tmdb_mapping,
    inputs="cleaned_links",
    outputs="movies_to_tmdb",
    name=create_tmdb_mapping.__name__,
)

index_ratings_node = node(
    func=index_ratings,
    inputs="cleaned_ratings",
    outputs=[
        "indexed_ratings",
        "movies_to_indices",
        "indices_to_movies",
        "users_to_indices",
    ],
    name=index_ratings.__name__,
)

normalize_ratings_node = node(
    func=normalize_ratings,
    inputs="indexed_ratings",
    outputs="normalized_ratings",
    name=normalize_ratings.__name__,
)

train_model_node = node(
    func=train_model,
    inputs=[
        "normalized_ratings",
        "params:training_split_ratio",
        "params:embedding_size",
        "params:learning_rate",
        "params:patience",
    ],
    outputs="trained_model",
    name=train_model.__name__,
)

recommend_movies_node = node(
    func=recommend_movies,
    inputs=[
        "trained_model",
        "cleaned_movies",
        "movies_to_tmdb",
        "normalized_ratings",
        "movies_to_indices",
        "indices_to_movies",
        "users_to_indices",
        "params:top_k",
        "params:user_ids"
    ],
    outputs="recommended_movies",
    name=recommend_movies.__name__,
)
