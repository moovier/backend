import pathlib

from typing import Union

from fastapi import FastAPI

app = FastAPI()

# GET /models -> [string] 
#   # ls "models" folder
#   # ["moovier_emb_10", "moovier_emb_25", "moovier_emb_50"]

# POST /predict (model_name: string) (input: [rows]) -> predictions
#    # take model name, inputs=(user_id) provided in api
#    # call recommend_movies_node with intermediate data from kedro run files /data/04_feature/...
#    # fetch movies for each movie id from tmdb api (#nice-to-have till due date)
#    # return result as json

# POST /train (model_name: string) (input: [row]) (expexted_output: [output]) -> [metric]
#   # call/invoke train_model_node kedro node which calls train_model function
#   # body needs user, movie, rating - validate user make sure it's within in the range as in data dir
#   # save the new model in models dir with new name so that 


@app.get("/models")
def read_models():
    clean_name = lambda n: n.name.removesuffix(".h5")
    models = pathlib.Path("../models/").glob("*.h5")
    return list(map(clean_name, models))
