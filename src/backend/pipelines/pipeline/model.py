import sys

import pandas as pd
import tensorflow as tf
import keras
from keras import layers, metrics, losses
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

    user_embedding = layers.Embedding(
        input_dim=num_users,
        output_dim=embedding_size,
        embeddings_initializer="he_normal",
        embeddings_regularizer=keras.regularizers.l2(1e-6),
    )(user_input)

    movie_embedding = layers.Embedding(
        input_dim=num_movies,
        output_dim=embedding_size,
        embeddings_initializer="he_normal",
        embeddings_regularizer=keras.regularizers.l2(1e-6),
    )(movie_input)

    user_bias = layers.Embedding(input_dim=num_users, output_dim=1)(user_input)
    movie_bias = layers.Embedding(input_dim=num_movies, output_dim=1)(movie_input)

    x = layers.Dot(axes=-1)([user_embedding, movie_embedding])
    x = layers.Add()([x, user_bias, movie_bias])
    x = layers.Dense(units=1, activation="relu", name="rating_prediction")(x)

    return keras.Model(inputs=[user_input, movie_input], outputs=x, name='recommender-net')


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
        loss=losses.MeanSquaredError(),
        metrics=metrics,
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
    )
    return model


def train_recommender(
    x: list[pd.Series],
    y: pd.Series,
    model: keras.Model,
    validation_split: float,
    patience: int,
) -> keras.Model:
    early_stopping_callback = keras.callbacks.EarlyStopping("val_loss", patience=patience)
    tensorboard_callback = keras.callbacks.TensorBoard("./data/08_reporting")
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath, "val_loss", save_best_only=True)

    model.fit(
        x=x,
        y=y,
        validation_split=validation_split,
        batch_size=64,
        epochs=sys.maxsize,
        callbacks=[early_stopping_callback, model_checkpoint_callback, tensorboard_callback],
    )

    return model
