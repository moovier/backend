import asyncio
import os
import pathlib
import re
from typing import Iterable

import http3
import keras
import pandas as pd
from aiocache import cached
from aiocache.serializers import PickleSerializer
from fastapi import Depends, FastAPI, HTTPException
from kedro.framework.context import KedroContext
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pydantic import BaseModel

from src.backend.pipelines.pipeline.nodes import (
    recommend_movies_node,
    train_model_node,
    normalize_ratings_node,
)


class TMDBClient:
    CACHE_TTL = 86400

    def __init__(self, api_key: str) -> None:
        self.api_key: str = api_key
        self.client = http3.AsyncClient()

    @cached(ttl=CACHE_TTL, serializer=PickleSerializer())
    async def fetch_movie(self, movie_id: str) -> dict:
        url = "https://api.themoviedb.org/3/movie/{}?api_key={}".format(movie_id, self.api_key)
        r = await self.client.get(url, verify=False)
        return r.json()

    async def fetch_movies(self, movie_ids: list[str]) -> list[dict]:
        result = await asyncio.gather(
            *[self.fetch_movie(id) for id in movie_ids],
            return_exceptions=True
        )
        return result

    @classmethod
    def get_instance(cls):
        api_key = os.environ.get('TMDB_API_KEY')
        yield cls(api_key)


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
async def predict(
    model_name: str,
    user_ids: list[int],
    top_k: int,
    context: KedroContext = Depends(get_context),
    tmdb_client: TMDBClient = Depends(TMDBClient.get_instance),
) -> dict[str, list[dict]]:
    if model_name not in list_models():
        raise HTTPException(status_code=404, detail="model not found")

    from_ctx = lambda x: context.catalog.load(x)
    result = recommend_movies_node.run(
        inputs={
            "trained_model": keras.models.load_model(f"models/{model_name}.h5"),
            "cleaned_movies": from_ctx("cleaned_movies"),
            "movies_to_tmdb": from_ctx("movies_to_tmdb"),
            "normalized_ratings": from_ctx("normalized_ratings"),
            "movies_to_indices": from_ctx("movies_to_indices"),
            "indices_to_movies": from_ctx("indices_to_movies"),
            "users_to_indices": from_ctx("users_to_indices"),
            "params:top_k": top_k,
            "params:user_ids": user_ids,
        }
    )

    mappings = {}
    for _, row in result["recommended_movies"].iterrows():
        user, rec = row.user, row.recommendations
        movie_ids = rec.split(",") if "," in rec else [rec]
        movies = await tmdb_client.fetch_movies(movie_ids)
        mappings[user] = movies

    return mappings


class DataFrame(BaseModel):
    users: list[int]
    movies: list[int]
    ratings: list[int]


@app.post("/train")
def train(
    model_name: str,
    validation_split: float,
    patience: float,
    body: DataFrame,
) -> str:
    if not model_name.replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="invalid model name")

    if len({len(l) for l in [body.users, body.movies, body.ratings]}) != 1:
        raise HTTPException(status_code=400, detail="incomplete dataset for training")

    if any(x not in [1, 2, 3, 4, 5] for x in body.ratings):
        raise HTTPException(status_code=400, detail="invalid ratings")

    dataset = pd.DataFrame({"user": body.users, "movie": body.movies, "rating": body.ratings})
    dataset = normalize_ratings_node.run(inputs={"indexed_ratings": dataset})

    trained_model = train_model_node.run(
        inputs={
            "built_model": keras.models.load_model(f"models/{model_name}.h5"),
            "normalized_ratings": dataset["normalized_ratings"],
            "params:validation_split": validation_split,
            "params:patience": patience
        }
    )

    new_model_name = update_model_name(model_name)
    pathlib.Path(f"models/{model_name}.h5").unlink()
    trained_model["trained_model"].save(f"models/{new_model_name}")

    return new_model_name


def update_model_name(model_name) -> str:
    match = re.search(r'_trained_(\d+)', model_name)
    prefix = re.sub(r'_\d+$', '', model_name)
    last_training = int(match.group(1)) if match else 0
    return f"{prefix}_{last_training + 1}.h5"
