"""
Shared vectorization logic — the single source of truth for turning a catalog
record OR a user profile into a 28-dim feature vector.

This module supplies the LOGIC; the fitted parameters (bounds, ingredient->
category map, category overrides, tag thresholds, default category preferences,
non-vegan ingredient list) are loaded from the model bundle (cbf_model.pkl) and
passed in. Meal vectors and user vectors MUST agree on every position or cosine
similarity is meaningless (handoff §5, §7).

Correctness is verified by tests/validate_skew.py, which rebuilds every catalog
meal vector with build_meal_vector() and asserts it equals the shipped
meal_matrix.npy. Until notebook3_feature_vectors.py is provided verbatim, that
golden test is what guarantees no train/serve skew on the meal side.
"""

import numpy as np

# --- The 28-position contract. Order is meaning; do not reorder. -------------
FEATURE_SCHEMA = [
    "calories_norm",    # 0
    "protein_ratio",    # 1
    "carb_ratio",       # 2
    "fat_ratio",        # 3
    "fiber_norm",       # 4
    "complexity_norm",  # 5
    "prep_time_norm",   # 6
    "is_breakfast",     # 7
    "is_lunch",         # 8
    "is_dinner",        # 9
    "is_snack",         # 10
    "is_vegetarian",    # 11
    "is_vegan",         # 12
    "is_high_protein",  # 13
    "is_high_carb",     # 14
    "is_low_fat",       # 15
    "is_high_fiber",    # 16
    "cat_grains",       # 17
    "cat_legumes",      # 18
    "cat_vegetables",   # 19
    "cat_fruits",       # 20
    "cat_poultry",      # 21
    "cat_fish",         # 22
    "cat_red_meat",     # 23
    "cat_dairy_eggs",   # 24
    "cat_nuts_seeds",   # 25
    "cat_fats_oils",    # 26
    "cat_other",        # 27
]

FOOD_CATEGORIES = [
    "grains", "legumes", "vegetables", "fruits", "poultry", "fish",
    "red_meat", "dairy_eggs", "nuts_seeds", "fats_oils", "other",
]

# Categories that disqualify a meal from being vegetarian.
_MEAT_CATEGORIES = {"poultry", "fish", "red_meat"}

_SLOT_INDEX = {"breakfast": 7, "lunch": 8, "dinner": 9, "snack": 10}
_CAT_OFFSET = 17  # first cat_* position


class OutOfVocabularyError(KeyError):
    """Raised when an ingredient is absent from the ingredient table / category
    map — we reject explicitly rather than silently defaulting (handoff §12)."""


# --- primitives ---------------------------------------------------------------

def macro_calorie_fractions(protein_g, carb_g, fat_g):
    """Protein/carb/fat as fractions of macro calories (4/4/9 kcal per g)."""
    pc, cc, fc = 4.0 * protein_g, 4.0 * carb_g, 9.0 * fat_g
    total = pc + cc + fc
    if total <= 0:
        return 0.0, 0.0, 0.0
    return pc / total, cc / total, fc / total


def ingredient_category_of(name, ingredient_category, category_overrides):
    """Internal 11-bucket category for an ingredient (override wins)."""
    if name in category_overrides:
        return category_overrides[name]
    if name in ingredient_category:
        return ingredient_category[name]
    raise OutOfVocabularyError(name)


def _aggregate(composition, ingredient_table):
    """Sum absolute macros over a composition + total grams."""
    totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0, "fiber": 0.0}
    grams = 0.0
    for item in composition:
        name = item["ingredient"]
        if name not in ingredient_table:
            raise OutOfVocabularyError(name)
        ing = ingredient_table[name]
        factor = item["grams"] / 100.0
        for k in totals:
            totals[k] += float(ing[k]) * factor
        grams += item["grams"]
    return totals, grams


def compute_category_proportions(composition, ingredient_table,
                                 ingredient_category, category_overrides):
    """Calorie fraction per food category, ordered by FOOD_CATEGORIES."""
    cal_by_cat = {c: 0.0 for c in FOOD_CATEGORIES}
    total_cal = 0.0
    for item in composition:
        name = item["ingredient"]
        ing = ingredient_table[name]
        cal = float(ing["calories"]) * item["grams"] / 100.0
        cat = ingredient_category_of(name, ingredient_category, category_overrides)
        cal_by_cat[cat] += cal
        total_cal += cal
    if total_cal <= 0:
        return [0.0] * len(FOOD_CATEGORIES)
    return [cal_by_cat[c] / total_cal for c in FOOD_CATEGORIES]


def derive_meal_tags(protein_ratio, carb_ratio, fat_ratio, fiber_per_100g,
                     categories_present, has_dairy_eggs, has_non_vegan,
                     tag_thresholds):
    is_vegetarian = 1.0 if categories_present.isdisjoint(_MEAT_CATEGORIES) else 0.0
    is_vegan = 1.0 if (is_vegetarian and not has_dairy_eggs and not has_non_vegan) else 0.0
    is_high_protein = 1.0 if protein_ratio >= tag_thresholds["high_protein"] else 0.0
    is_high_carb = 1.0 if carb_ratio >= tag_thresholds["high_carb"] else 0.0
    is_low_fat = 1.0 if fat_ratio <= tag_thresholds["low_fat"] else 0.0
    is_high_fiber = 1.0 if fiber_per_100g >= tag_thresholds["high_fiber"] else 0.0
    return (is_vegetarian, is_vegan, is_high_protein,
            is_high_carb, is_low_fat, is_high_fiber)


# --- meal side ----------------------------------------------------------------

def build_meal_vector(record, ingredient_table, params):
    """catalog record -> 28-dim np.float32 vector."""
    bounds = params["bounds"]
    ingredient_category = params["ingredient_category"]
    category_overrides = params.get("category_overrides", {})
    non_vegan = set(params.get("non_vegan_ingredients", []))
    tag_thresholds = params["tag_thresholds"]

    totals, grams = _aggregate(record["composition"], ingredient_table)
    per100 = {k: (v / grams * 100.0 if grams else 0.0) for k, v in totals.items()}
    pr, cr, fr = macro_calorie_fractions(totals["protein"], totals["carbs"], totals["fat"])

    v = [0.0] * len(FEATURE_SCHEMA)
    v[0] = per100["calories"] / bounds["max_calories"]
    v[1], v[2], v[3] = pr, cr, fr
    v[4] = per100["fiber"] / bounds["max_fiber"]
    v[5] = record["complexity"] / 10.0
    v[6] = record["prep_minutes"] / bounds["max_prep"]

    meal_types = set(record["meal_type"])
    v[7] = 1.0 if "breakfast" in meal_types else 0.0
    v[8] = 1.0 if "lunch" in meal_types else 0.0
    v[9] = 1.0 if "dinner" in meal_types else 0.0
    v[10] = 1.0 if "snack" in meal_types else 0.0

    categories_present = set()
    has_dairy_eggs = False
    has_non_vegan = False
    for item in record["composition"]:
        cat = ingredient_category_of(item["ingredient"], ingredient_category, category_overrides)
        categories_present.add(cat)
        if cat == "dairy_eggs":
            has_dairy_eggs = True
        if item["ingredient"] in non_vegan:
            has_non_vegan = True

    v[11:17] = derive_meal_tags(pr, cr, fr, per100["fiber"], categories_present,
                                has_dairy_eggs, has_non_vegan, tag_thresholds)
    v[_CAT_OFFSET:] = compute_category_proportions(
        record["composition"], ingredient_table, ingredient_category, category_overrides)
    return np.asarray(v, dtype=np.float32)


# --- user side ----------------------------------------------------------------
# PROVISIONAL until notebook3_feature_vectors.py is confirmed. The magnitude dims
# (0 calories_norm, 4 fiber_norm, 6 prep_time_norm) are left at 0: they are not
# user *preferences* — calorie targeting is handled by post-process portion
# scaling, and there is no user-side fibre/prep preference (handoff §5 notes
# is_high_fiber is always 0 on the user side). All *preference* dims are set.

def build_user_vector(target_calories, target_protein_pct, target_carb_pct,
                      target_fat_pct, complexity_target, dietary_lifestyle,
                      meal_slot, bounds, default_cat_prefs, tag_thresholds,
                      preferred_categories=None):
    v = [0.0] * len(FEATURE_SCHEMA)

    v[1], v[2], v[3] = target_protein_pct, target_carb_pct, target_fat_pct
    v[5] = complexity_target / 10.0

    if meal_slot not in _SLOT_INDEX:
        raise ValueError(f"unknown meal_slot {meal_slot!r}")
    v[_SLOT_INDEX[meal_slot]] = 1.0

    is_veg = dietary_lifestyle in ("vegetarian", "vegan")
    v[11] = 1.0 if is_veg else 0.0
    v[12] = 1.0 if dietary_lifestyle == "vegan" else 0.0
    v[13] = 1.0 if (dietary_lifestyle == "high_protein"
                    or target_protein_pct >= tag_thresholds["high_protein"]) else 0.0
    v[14] = 1.0 if target_carb_pct >= tag_thresholds["high_carb"] else 0.0
    v[15] = 1.0 if target_fat_pct <= tag_thresholds["low_fat"] else 0.0
    v[16] = 0.0  # user is_high_fiber always 0 (handoff §5)

    prefs = (preferred_categories
             or default_cat_prefs.get(dietary_lifestyle)
             or default_cat_prefs["omnivore"])
    for i, cat in enumerate(FOOD_CATEGORIES):
        v[_CAT_OFFSET + i] = float(prefs.get(cat, 0.0))
    return np.asarray(v, dtype=np.float32)


# --- filtering + ranking ------------------------------------------------------

def get_excluded_ingredients(allergens, allergen_map):
    excluded = set()
    for allergen in (allergens or []):
        excluded.update(allergen_map.get(allergen, []))
    return excluded


def build_eligible_pool(catalog, ingredient_table, allergen_map, allergens=None,
                        dislikes=None, exclude_meal_ids=None, complexity_max=10,
                        lifestyle=None, meal_flags=None):
    """Return eligible catalog indices after hard filters (run BEFORE ranking).

    Dietary lifestyle is treated as a HARD filter for vegan/vegetarian (a vegan
    plan must never contain animal products) — a deliberate deviation from the
    handoff, which modelled lifestyle only as a soft vector preference. Requires
    meal_flags[idx] = {"is_vegetarian": bool, "is_vegan": bool}. omnivore /
    high_protein impose no lifestyle filter.
    """
    excluded_ing = get_excluded_ingredients(allergens, allergen_map)
    dislikes_lower = {d.lower() for d in (dislikes or [])}
    exclude_ids = set(exclude_meal_ids or [])

    eligible = []
    for idx, record in enumerate(catalog):
        if idx in exclude_ids:
            continue
        if record["complexity"] > complexity_max:
            continue
        if record["name"].lower() in dislikes_lower:
            continue
        ingredients = {c["ingredient"] for c in record["composition"]}
        if ingredients & excluded_ing:
            continue
        if ingredients & dislikes_lower:  # dislike by ingredient too
            continue
        if lifestyle in ("vegan", "vegetarian") and meal_flags is not None:
            flags = meal_flags[idx]
            if lifestyle == "vegan" and not flags["is_vegan"]:
                continue
            if lifestyle == "vegetarian" and not flags["is_vegetarian"]:
                continue
        eligible.append(idx)
    return eligible


def rank_meals(user_vector, meal_matrix, eligible_ids, meal_names, top_n=20):
    """Cosine-rank eligible meals against the user vector."""
    u = np.asarray(user_vector, dtype=np.float32)
    u_norm = float(np.linalg.norm(u)) or 1e-9
    scored = []
    for idx in eligible_ids:
        m = meal_matrix[idx]
        m_norm = float(np.linalg.norm(m)) or 1e-9
        score = float(np.dot(u, m) / (u_norm * m_norm))
        scored.append({"id": idx, "name": meal_names[idx], "score": score})
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:top_n] if top_n else scored
