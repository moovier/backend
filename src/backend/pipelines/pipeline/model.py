import sys

import tensorflow as tf
import keras
from keras.optimizers import Adam
from keras import layers
from kedro.config import ConfigLoader


def build_recommender_model(num_users, num_movies, embedding_size):

    input_pair = keras.Input(shape=(2,), name="user_movie_input", dtype=tf.int32)
    user_input = layers.Lambda(lambda x: x[:, 0])(input_pair)
    movie_input = layers.Lambda(lambda x: x[:, 1])(input_pair)

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

    dot_user_movie = layers.Dot(axes=-1)([user_embedding, movie_embedding])
    x = layers.Add()([dot_user_movie, user_bias, movie_bias])

    output = layers.Activation("sigmoid")(x)

    return keras.Model(inputs=input_pair, outputs=output, name='recommender-net')


def train_recommender(
    x,
    y,
    num_users,
    num_movies,
    embedding_size,
    learning_rate,
    train_split,
    patience
):
    x_train, x_val = x[:train_split], x[train_split:]
    y_train, y_val = y[:train_split], y[train_split:]

    model = build_recommender_model(
        num_users=num_users,
        num_movies=num_movies,
        embedding_size=embedding_size,
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["binary_crossentropy"]
    )

    filepath = (
        ConfigLoader(conf_source="./conf")
        .get("catalog.yml")
        .get("trained_model")
        .get("filepath")
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            save_best_only=True,
            monitor="val_loss"
        ),
        keras.callbacks.TensorBoard(
            log_dir="./data/08_reporting"
        )
    ]

    model.fit(
        x=x_train,
        y=y_train,
        batch_size=64,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        epochs=sys.maxsize  # in callbacks, we trust
    )

    return model
