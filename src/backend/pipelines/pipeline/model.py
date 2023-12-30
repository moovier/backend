import sys

import pandas as pd
import tensorflow as tf
import keras
from keras import layers, metrics, losses
from keras.utils import plot_model
from kedro.config import ConfigLoader

filepath = (
    ConfigLoader(conf_source="./conf")
    .get("catalog.yml")
    .get("trained_model")
    .get("filepath")
)

metrics = [
    metrics.MeanSquaredError(),
    metrics.MeanAbsoluteError(),
    metrics.RootMeanSquaredError(),
]


def build_recommender_model(num_users: int, num_movies: int, embedding_size: int) -> keras.Model:
    user_input = layers.Input(shape=(1,), name="user_input", dtype=tf.int32)
    movie_input = layers.Input(shape=(1,), name="movie_input", dtype=tf.int32)

    movie_embedding_mlp = layers.Embedding(num_movies, embedding_size, name="movie_embedding_mlp")(movie_input)
    movie_vec_mlp = layers.Flatten(name="flatten_movie_mlp")(movie_embedding_mlp)

    user_embedding_mlp = layers.Embedding(num_users, embedding_size, name="user_embedding_mlp")(user_input)
    user_vec_mlp = layers.Flatten(name="flatten_user_mlp")(user_embedding_mlp)

    movie_embedding_mf = layers.Embedding(num_movies, embedding_size, name="movie_embedding_mf")(movie_input)
    movie_vec_mf = layers.Flatten(name="flatten_movie_mf")(movie_embedding_mf)

    user_embedding_mf = layers.Embedding(num_users, embedding_size, name="user_embedding_mf")(user_input)
    user_vec_mf = layers.Flatten(name="flatten_user_mf")(user_embedding_mf)

    concat = layers.Concatenate()([movie_vec_mlp, user_vec_mlp])
    pred_mlp = layers.Dense(units=10, name="pred_mlp", activation="relu")(concat)
    pred_mf = layers.Dot(axes=-1, name="pred_mf")([movie_vec_mf, user_vec_mf])
    combine_mlp_mf = layers.Concatenate()([pred_mf, pred_mlp])

    result = layers.Dense(1, name="result", activation="relu")(combine_mlp_mf)
    return keras.Model(inputs=[user_input, movie_input], outputs=result, name="recommender-net")


def build_recommender(
    num_users: int,
    num_movies: int,
    embedding_size: int,
    learning_rate: float,
) -> keras.Model:

    model = build_recommender_model(
        num_users=num_users,
        num_movies=num_movies,
        embedding_size=embedding_size,
    )

    model.compile(
        loss=losses.MeanAbsoluteError(),
        metrics=metrics,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    )
    return model


def train_recommender(
    x: list[pd.Series],
    y: pd.Series,
    model: keras.Model,
    validation_split: float,
    patience: int,
    id: str,
) -> keras.Model:

    early_stopping_callback = keras.callbacks.EarlyStopping("val_loss", patience=patience)
    tensorboard_callback = keras.callbacks.TensorBoard(f"./data/08_reporting/{id}/tensorboard")
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath, "val_loss", save_best_only=True)

    model.fit(
        x=x,
        y=y,
        validation_split=validation_split,
        batch_size=64,
        epochs=sys.maxsize,
        callbacks=[early_stopping_callback, model_checkpoint_callback, tensorboard_callback],
    )

    plot_model(
        model,
        to_file=f"./data/08_reporting/{id}/{model.name}.png",
        show_layer_names=True,
        show_layer_activations=True,
    )

    return model
