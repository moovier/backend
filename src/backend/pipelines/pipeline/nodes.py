import numpy as np
import pandas as pd
from kedro.pipeline import node

from .model import train_recommender


def drop_year_and_translation(movies):
    def clean_title(title):
        title_clear = title.split(" (")[0]
        parts = title_clear.split(", ")
        return " ".join(parts[::-1]) if len(parts) > 1 else title_clear

    movies["title"] = movies["title"].apply(clean_title)
    return movies


def drop_genres(movies):
    return movies.drop(columns="genres")


def drop_timestamp(ratings):
    return ratings.drop(columns="timestamp")


def index_ratings(ratings):
    def create_index_maps(x):
        x = x.unique().tolist()
        idx_map = dict(enumerate(x))
        rev_map = dict(map(reversed, idx_map.items()))
        return idx_map, rev_map

    _, users_to_indices = create_index_maps(ratings.userId)
    indices_to_movies, movies_to_indices = create_index_maps(ratings.movieId)

    ratings["user"] = ratings.userId.map(users_to_indices)
    ratings["movie"] = ratings.movieId.map(movies_to_indices)
    ratings["rating"] = ratings.rating.values.astype(np.float32)

    return ratings, movies_to_indices, indices_to_movies, users_to_indices


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
    ratings,
    movies_to_indices,
    indices_to_movies,
    users_to_indices,
    top_k,
    num_users,
):
    recommended_movies_for_all_users = []

    for user_id in ratings["userId"].unique()[:num_users]:
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
        ratings_pred = ratings_pred.flatten()
        ratings_sorted_indices = ratings_pred.argsort()[-top_k:][::-1]

        recommended_movies = [
            indices_to_movies.get(movies_not_watched[rating][0])
            for rating in ratings_sorted_indices
        ]
        recommended_movies = movies[movies["movieId"].isin(recommended_movies)]["title"]

        recommended_movies_for_all_users.append(
            {"userId": user_id, "recommendations": "|".join(recommended_movies)}
        )

    return pd.DataFrame(recommended_movies_for_all_users)


drop_year_and_translation_node = node(
    func=drop_year_and_translation,
    inputs="movies",
    outputs="cleaned_title",
    name=drop_year_and_translation.__name__,
)

drop_genres_node = node(
    func=drop_genres,
    inputs="cleaned_title",
    outputs="cleaned_movies",
    name=drop_genres.__name__,
)

drop_timestamp_node = node(
    func=drop_timestamp,
    inputs="ratings",
    outputs="cleaned_ratings",
    name=drop_timestamp.__name__,
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
        "normalized_ratings",
        "movies_to_indices",
        "indices_to_movies",
        "users_to_indices",
        "params:top_k",
        "params:num_users",
    ],
    outputs="recommended_movies",
    name=recommend_movies.__name__,
)
