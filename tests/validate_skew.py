"""
Golden train/serve-skew guard: rebuild every catalog meal vector with the
shared build_meal_vector() and assert it reproduces the shipped meal_matrix.npy.
If this passes, the service's meal-side vectorization is provably identical to
the notebook that produced the artifacts (handoff §7, §12).

Run:  ../.venv/bin/python validate_skew.py   (from tests/), or
      .venv/bin/python tests/validate_skew.py (from recommendation-service/)
"""

import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import vectorization as vz  # noqa: E402

ARTIFACTS = os.environ.get(
    "ARTIFACTS_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "track-treat", "artifacts")),
)

ATOL = 1e-3  # meal_matrix is stored float32


def main():
    bundle = pickle.load(open(os.path.join(ARTIFACTS, "cbf_model.pkl"), "rb"))
    catalog = json.load(open(os.path.join(ARTIFACTS, "meal_catalog.json")))
    ingredient_table = json.load(open(os.path.join(ARTIFACTS, "ingredient_table.json")))
    matrix = np.load(os.path.join(ARTIFACTS, "meal_matrix.npy"))

    assert bundle["feature_schema"] == vz.FEATURE_SCHEMA, "feature_schema mismatch"
    assert bundle["food_categories"] == vz.FOOD_CATEGORIES, "food_categories mismatch"
    assert matrix.shape == (len(catalog), len(vz.FEATURE_SCHEMA)), "shape mismatch"

    params = {
        "bounds": bundle["bounds"],
        "ingredient_category": bundle["ingredient_category"],
        "category_overrides": bundle.get("category_overrides", {}),
        "non_vegan_ingredients": bundle.get("non_vegan_ingredients", []),
        "tag_thresholds": bundle["tag_thresholds"],
    }

    rebuilt = np.vstack([vz.build_meal_vector(rec, ingredient_table, params) for rec in catalog])
    diff = np.abs(rebuilt - matrix)

    max_err = float(diff.max())
    n_bad_rows = int((diff.max(axis=1) > ATOL).sum())
    print(f"meals: {len(catalog)}   max abs error: {max_err:.6f}   rows over tol: {n_bad_rows}")

    if n_bad_rows:
        # Per-feature worst offenders to guide any fix.
        per_feat = diff.max(axis=0)
        worst = np.argsort(per_feat)[::-1][:6]
        print("worst features:")
        for j in worst:
            print(f"  [{j:2d}] {vz.FEATURE_SCHEMA[j]:16s} max err {per_feat[j]:.4f}")
        bad = np.where(diff.max(axis=1) > ATOL)[0][:3]
        for i in bad:
            js = np.where(diff[i] > ATOL)[0]
            print(f"  meal {i} '{bundle['meal_names'][i]}' bad dims {js.tolist()}")
            for j in js:
                print(f"      [{j}] {vz.FEATURE_SCHEMA[j]}: got {rebuilt[i][j]:.4f} want {matrix[i][j]:.4f}")
        print("SKEW TEST: FAIL")
        sys.exit(1)

    print("SKEW TEST: PASS — meal-side vectorization matches the shipped matrix exactly.")


if __name__ == "__main__":
    main()
