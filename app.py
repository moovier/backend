import pathlib
import keras
from typing import Any, Iterable, Union, Annotated

from fastapi import Depends, FastAPI, HTTPException
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
def list_models() -> list[str]:
    models = pathlib.Path("models/").glob("*.h5")
    return [model.stem for model in models]


@app.post("/predict")
def predict(
    model_name: str,
    user_ids: list[int],
    top_k: int,
    session: KedroSession = Depends(get_session),
    context: KedroContext = Depends(get_context),
) -> dict[str, str]:
    if model_name not in list_models():
        raise HTTPException(status_code=404, detail="model not found")

    model = keras.models.load_model(f"models/{model_name}.h5")

    session.run("pipeline", 
        node_names=["recommend_movies"], 
        from_inputs={"trained_model": model, "params:user_ids": user_ids, "params:top_k": top_k},
    )

    return context.catalog.load("recommended_movies").to_dict()["recommendations"]


@app.post("/train")
def train(
    old_model_name: str,
    new_model_name: str,
    training_split_ratio: float,
    embedding_size: float, 
    learning_rate: float,
    patience: float,
    ratings: list,
    session: KedroSession = Depends(get_session),
    context: KedroContext = Depends(get_context),
) -> str:
    if not old_model_name.replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="invalid model name")

    session.run("pipeline", 
        node_names=["train_model"],
        from_inputs={
            "params:training_split_ratio": training_split_ratio,
            "params:embedding_size": embedding_size,
            "params:learning_rate": learning_rate,
            "params:patience": patience,
        },
    )

    model = context.catalog.load("trained_model")
    model.save(f"models/{new_model_name}.h5")
    return new_model_name
