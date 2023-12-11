import pathlib
from typing import Any, Iterable, Union

from fastapi import Depends, FastAPI
from kedro.framework.context import KedroContext
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


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



app = FastAPI(
    title="FastAPI + Kedro",
    version="0.0.1",
    license_info={
        "name": "GNU GENERAL PUBLIC LICENSE",
        "url": "https://www.gnu.org/licenses/gpl-3.0.html",
    },
)


def get_session() -> Iterable[KedroSession]:
    bootstrap_project(pathlib.Path().cwd())
    with KedroSession.create() as session:
        yield session


def get_context(session: KedroSession = Depends(get_session)) -> Iterable[KedroContext]:
    yield session.load_context()


@app.get("/models")
def list_models():
    models = pathlib.Path("../models/").glob("*.h5")
    return [model.stem for model in models]


@app.post("/predict")
def predict(
    session: KedroSession = Depends(get_session),
    context: KedroContext = Depends(get_context),
) -> dict[str, Any]:
    session.run("pipeline")
    catalog = context.catalog
    return catalog.load("output")


@app.post("/train")
def train():
    return
