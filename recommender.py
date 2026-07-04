"""
MealRecommender: loads the frozen model bundle + catalog + ingredient table on
startup and turns a {userProfile, adaptiveProfile} request into a 7-day plan.

The recommendation path reads ONLY the artifacts (pkl/npy + catalog + ingredient
table) — never the DB — so a plan is a pure function of a versioned artifact set
(handoff §4).
"""

import hashlib
import json
import os
import pickle

import numpy as np

import profile_mapping as pm
import vectorization as vz
from config import (ARTIFACTS_DIR, CALORIE_TOLERANCE, DEFAULT_COMPLEXITY_MAX,
                    MAX_REPEATS_PER_WEEK, PREFERRED_BLEND_WEIGHT)
from postprocess import assemble_plan


class MealRecommender:
    def __init__(self, artifacts_dir=ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self.ready = False
        self.warnings = []
        self._load()

    # --- loading --------------------------------------------------------------

    def _load_bundle(self, path):
        """D3: single swappable entry point for the serialization format."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load(self):
        pkl_path = os.path.join(self.artifacts_dir, "cbf_model.pkl")
        b = self._load_bundle(pkl_path)

        self.meal_matrix = np.asarray(b["meal_matrix"], dtype=np.float32)
        self.meal_ids = list(b["meal_ids"])
        self.meal_names = list(b["meal_names"])
        self.feature_schema = list(b["feature_schema"])
        self.bounds = dict(b["bounds"])
        self.allergen_map = dict(b["allergen_map"])
        self.default_cat_prefs = dict(b["default_cat_prefs"])
        self.tag_thresholds = dict(b["tag_thresholds"])
        self.params = {
            "bounds": self.bounds,
            "ingredient_category": dict(b["ingredient_category"]),
            "category_overrides": dict(b.get("category_overrides", {})),
            "non_vegan_ingredients": list(b.get("non_vegan_ingredients", [])),
            "tag_thresholds": self.tag_thresholds,
        }

        # Skew / consistency guards (handoff §4, §7).
        assert self.feature_schema == vz.FEATURE_SCHEMA, "feature_schema mismatch (train/serve skew)"
        assert b["food_categories"] == vz.FOOD_CATEGORIES, "food_categories mismatch"
        assert self.meal_matrix.shape == (len(self.meal_ids), len(vz.FEATURE_SCHEMA)), "matrix shape mismatch"

        self.artifact_version = self._version(pkl_path)
        self.catalog = json.load(open(os.path.join(self.artifacts_dir, "meal_catalog.json")))
        self.ingredient_table = json.load(open(os.path.join(self.artifacts_dir, "ingredient_table.json")))
        assert len(self.catalog) == len(self.meal_ids), "catalog / meal_ids length mismatch"

        self._check_vocab()
        self.meal_info = self._enrich()
        self.id_to_row = {mid: i for i, mid in enumerate(self.meal_ids)}
        # is_vegetarian (dim 11) / is_vegan (dim 12) per meal, for hard lifestyle filtering.
        veg_col = vz.FEATURE_SCHEMA.index("is_vegetarian")
        vegan_col = vz.FEATURE_SCHEMA.index("is_vegan")
        self.meal_flags = [
            {"is_vegetarian": bool(row[veg_col] >= 0.5), "is_vegan": bool(row[vegan_col] >= 0.5)}
            for row in self.meal_matrix
        ]
        self.ready = True

    def _version(self, path):
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
        return f"cbf-{digest}"

    def _check_vocab(self):
        table = set(self.ingredient_table)
        missing = {c["ingredient"] for rec in self.catalog for c in rec["composition"]
                   if c["ingredient"] not in table}
        if missing:
            self.warnings.append(f"catalog ingredients missing from ingredient_table: {sorted(missing)}")

    def _enrich(self):
        """Precompute per-meal composition + absolute nutrition for one serving."""
        info = {}
        for idx, rec in enumerate(self.catalog):
            ingredients = []
            totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0, "fiber": 0.0}
            for c in rec["composition"]:
                ing = self.ingredient_table[c["ingredient"]]
                factor = c["grams"] / 100.0
                ingredients.append({
                    "foodName": c["ingredient"],
                    "grams": float(c["grams"]),
                    "calories": round(ing["calories"] * factor, 1),
                    "protein": round(ing["protein"] * factor, 2),
                    "carbs": round(ing["carbs"] * factor, 2),
                    "fat": round(ing["fat"] * factor, 2),
                })
                for k in totals:
                    totals[k] += ing[k] * factor
            info[idx] = {
                "recipeId": self.meal_ids[idx],
                "name": self.meal_names[idx],
                "mealTypes": rec["meal_type"],
                "prepMinutes": rec["prep_minutes"],
                "complexity": rec["complexity"],
                "ingredients": ingredients,
                "calories": round(totals["calories"], 1),
                "protein": round(totals["protein"], 2),
                "carbs": round(totals["carbs"], 2),
                "fat": round(totals["fat"], 2),
                "fiber": round(totals["fiber"], 2),
            }
        return info

    # --- serving --------------------------------------------------------------

    def health(self):
        return {
            "status": "ready" if self.ready else "loading",
            "nMeals": len(self.meal_ids),
            "featureDims": len(self.feature_schema),
            "artifactVersion": self.artifact_version,
            "warnings": self.warnings,
        }

    def _preferred_centroid(self, preferred_ids):
        rows = [self.id_to_row[i] for i in (preferred_ids or []) if i in self.id_to_row]
        return self.meal_matrix[rows].mean(axis=0) if rows else None

    @staticmethod
    def _blend(user_vector, centroid, weight):
        def unit(x):
            n = np.linalg.norm(x)
            return x / n if n else x
        return (1.0 - weight) * unit(user_vector) + weight * unit(centroid)

    def generate(self, user_profile, adaptive_profile=None):
        if not self.ready:
            raise RuntimeError("recommender not ready")
        up = user_profile or {}
        ap = adaptive_profile or {}

        lifestyle = pm.normalize_lifestyle(up.get("dietaryLifestyle", "omnivore"))
        macros = up.get("macroTargets") or {}
        target_calories = float(up.get("targetCalories") or 2000)
        protein_pct = float(macros.get("protein", 0.30))
        carb_pct = float(macros.get("carb", macros.get("carbs", 0.40)))
        fat_pct = float(macros.get("fat", 0.30))

        slots = pm.active_slots(up)
        allergens = pm.validate_allergens(up.get("allergies"), self.allergen_map)
        dislikes = up.get("dislikes") or []

        complexity_target = int(ap.get("complexityTarget") or DEFAULT_COMPLEXITY_MAX)
        exclude = ap.get("skippedRecipes") or ap.get("skippedFoods") or []
        preferred = ap.get("preferredRecipes") or []
        slot_adherence = ap.get("slotAdherence") or {}

        # Hard filters once, before ranking (handoff §8).
        eligible = vz.build_eligible_pool(
            self.catalog, self.ingredient_table, self.allergen_map,
            allergens=allergens, dislikes=dislikes,
            exclude_meal_ids=exclude, complexity_max=complexity_target,
            lifestyle=lifestyle, meal_flags=self.meal_flags,
        )
        if not eligible:
            raise ValueError("no eligible meals after filtering (over-constrained profile)")

        centroid = self._preferred_centroid(preferred)

        ranked_by_slot = {}
        for slot in set(slots):
            uv = vz.build_user_vector(
                target_calories, protein_pct, carb_pct, fat_pct,
                complexity_target, lifestyle, slot,
                self.bounds, self.default_cat_prefs, self.tag_thresholds,
            )
            if centroid is not None:
                uv = self._blend(uv, centroid, PREFERRED_BLEND_WEIGHT)
            ranked = vz.rank_meals(uv, self.meal_matrix, eligible, self.meal_names, top_n=None)
            ranked_by_slot[slot] = [r["id"] for r in ranked]

        days = assemble_plan(
            slots, ranked_by_slot, self.meal_info, target_calories,
            slot_adherence=slot_adherence, calorie_tolerance=CALORIE_TOLERANCE,
            max_repeats=MAX_REPEATS_PER_WEEK,
        )
        return {"artifactVersion": self.artifact_version, "days": days}
