# Moovier - Movie Recommendation System

Moovier is a novel movie recommendation system that uses [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering) to offer personalized movie suggestions.

### Installation

To use this application, an [TMDB API](https://www.themoviedb.org/) key is required to fetch information regarding the recommended movies.

   ```bash
   git clone https://github.com/moovier/backend.git     # clone back-end
   git clone https://github.com/moovier/frontend.git    # clone front-end
  
   python -m venv moovier                               # create moovier venv
   source moovier/bin/activate                          # activate moovier

   cd moovier/backend                                   # navigate to back-end
   pip install -r src/requirements.txt                  # install dependencies
   export TMDB_API_KEY=secret-key                       # set-up tmdb key
   
   cd ../frontend                                       # navigate to front-end
   npm install                                          # install dependencies
   ```

### Usage

The project can be either run as a standalone [Kedro](https://kedro.org/) pipeline using `kedro run` or as
a back-end API using `uvicorn app:app`. The application exposes the following endpoints:

#### API

 - `/models` serves the names of pre-trained models that can be used for inference or fine-tuning.
    ```bash
   curl -X 'GET' \
   'http://127.0.0.1:8000/models' \
   -H 'accept: application/json'
   ```
 - `/predict` is used for returning movie recommendations. `model_name` specifies which of the pre-trained models to use for inference, while `top_k` selects how many movies to return for each `user_id` passed in the body.
   ```bash
   curl -X 'POST' \
   'http://127.0.0.1:8000/predict?model_name=moovier_emb_25_trained_0&top_k=1' \
   -H 'accept: application/json' \
   -H 'Content-Type: application/json' \
   -d '[1, 2, 3, 4]'
   ```

 - `/train` fine-tunes the selected `model_name` with a new batch of data. `validation_split` can be used to create a validation dataset for evaluation purposes; `patience` specifies at which point to activate the early stopping callback. By default, the `val_loss` is being monitored. `ratings` is the incoming batch of data on top of which the model is fine-tuned.
   ```bash
   curl -X 'POST' \
   'http://127.0.0.1:8000/train?model_name=moovier_emb_25_trained_0&validation_split=0.1&patience=3' \
   -H 'accept: application/json' \
   -H 'Content-Type: multipart/form-data' \
   -F 'ratings=string'
   ```

#### Kedro
To run only prycaret nodes
```bash
kedro run --nodes=pycaret_predict_ratings 
```

