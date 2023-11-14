import tensorflow as tf
from keras.optimizers import Adam
from tensorflow import keras
from keras import layers


def build_recommender_model(num_users, num_movies, embedding_size):
    # Input layer
    input_pair = keras.Input(shape=(2,), name="user_movie_input", dtype=tf.int32)

    # Split the input into separate user and movie tensors
    user_input = layers.Lambda(lambda x: x[:, 0])(input_pair)
    movie_input = layers.Lambda(lambda x: x[:, 1])(input_pair)

    # Embedding layers
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

    # Bias layers
    user_bias = layers.Embedding(input_dim=num_users, output_dim=1)(user_input)
    movie_bias = layers.Embedding(input_dim=num_movies, output_dim=1)(movie_input)

    # Dot product and summation
    dot_user_movie = layers.Dot(axes=-1)([user_embedding, movie_embedding])
    x = layers.Add()([dot_user_movie, user_bias, movie_bias])

    # Output layer
    output = layers.Activation("sigmoid")(x)

    # Build the model
    model = keras.Model(inputs=input_pair, outputs=output, name='recommender-net')
    return model


def train_recommender(
        x, y,
        num_users, num_movies,
        embedding_size,
        learning_rate,
        train_split
):
    x_train = x[:train_split]
    x_val = x[train_split:]
    y_train = y[:train_split]
    y_val = y[train_split:]

    model = build_recommender_model(
        num_users=num_users,
        num_movies=num_movies,
        embedding_size=embedding_size,
        # name='recommender-net'
    )

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
    )

    model.fit(
        x=x_train,
        y=y_train,
        batch_size=64,
        epochs=1,
        verbose=1,
        validation_data=(x_val, y_val),
    )

    return model
