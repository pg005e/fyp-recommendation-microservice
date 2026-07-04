import json
import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ARTIFACTS = os.environ.get(
    "ARTIFACTS_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "track-treat", "artifacts")),
)


@pytest.fixture(scope="session")
def bundle():
    return pickle.load(open(os.path.join(ARTIFACTS, "cbf_model.pkl"), "rb"))


@pytest.fixture(scope="session")
def catalog():
    return json.load(open(os.path.join(ARTIFACTS, "meal_catalog.json")))


@pytest.fixture(scope="session")
def ingredient_table():
    return json.load(open(os.path.join(ARTIFACTS, "ingredient_table.json")))


@pytest.fixture(scope="session")
def matrix():
    return np.load(os.path.join(ARTIFACTS, "meal_matrix.npy"))


@pytest.fixture(scope="session")
def recommender():
    from recommender import MealRecommender
    return MealRecommender(ARTIFACTS)
