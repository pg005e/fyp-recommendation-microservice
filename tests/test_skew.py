"""Train/serve skew: rebuilt meal vectors must equal the shipped matrix."""

import numpy as np

import vectorization as vz

ATOL = 1e-3  # matrix is stored float32


def _params(bundle):
    return {
        "bounds": bundle["bounds"],
        "ingredient_category": bundle["ingredient_category"],
        "category_overrides": bundle.get("category_overrides", {}),
        "non_vegan_ingredients": bundle.get("non_vegan_ingredients", []),
        "tag_thresholds": bundle["tag_thresholds"],
    }


def test_schema_matches_bundle(bundle, matrix):
    assert bundle["feature_schema"] == vz.FEATURE_SCHEMA
    assert bundle["food_categories"] == vz.FOOD_CATEGORIES
    assert matrix.shape == (len(bundle["meal_ids"]), len(vz.FEATURE_SCHEMA))


def test_meal_vectors_reproduce_matrix(bundle, catalog, ingredient_table, matrix):
    params = _params(bundle)
    rebuilt = np.vstack([vz.build_meal_vector(r, ingredient_table, params) for r in catalog])
    assert np.abs(rebuilt - matrix).max() < ATOL


def test_user_vector_length_and_schema(bundle):
    uv = vz.build_user_vector(2000, 0.3, 0.4, 0.3, 5, "omnivore", "lunch",
                              bundle["bounds"], bundle["default_cat_prefs"], bundle["tag_thresholds"])
    assert uv.shape == (len(vz.FEATURE_SCHEMA),)
    # user & meal vectors share length -> cosine is well-defined
    assert uv.shape[0] == matrix_dim(bundle)


def matrix_dim(bundle):
    return len(bundle["feature_schema"])
