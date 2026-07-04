"""
Validate + normalize the request profile into vector inputs.

NestJS owns the semantic mapping (dietaryLifestyle enum -> vector lifestyle,
free-text allergies -> allergen ids) per the plan (D1/D4). This module is the
defensive boundary: it rejects unknown values EXPLICITLY rather than silently
defaulting to omnivore / an empty allergen set (handoff gotchas #1, §8, §12).
"""

VECTOR_LIFESTYLES = {"omnivore", "vegetarian", "vegan", "high_protein"}

_BASE_SLOTS = ["breakfast", "lunch", "dinner"]


def normalize_lifestyle(value):
    if value not in VECTOR_LIFESTYLES:
        raise ValueError(
            f"dietaryLifestyle {value!r} is not a vector lifestyle {sorted(VECTOR_LIFESTYLES)}. "
            "NestJS must map it before calling (do not rely on a silent fallback)."
        )
    return value


def validate_allergens(allergens, allergen_map):
    """Every allergen id must be known; unknown ids are a hard error, not a
    no-op exclusion (the silent-empty-set bug the handoff warns about)."""
    unknown = [a for a in (allergens or []) if a not in allergen_map]
    if unknown:
        raise ValueError(f"unknown allergen ids {unknown}; known: {sorted(allergen_map)}")
    return list(allergens or [])


def derive_slots(meals_per_day):
    """Fallback slot structure from an integer meals/day (NestJS normally sends
    an explicit `slots` list). 2 -> lunch/dinner, 3 -> +breakfast, 4 -> +snack,
    >4 -> extra snacks."""
    n = int(meals_per_day or 3)
    if n <= 2:
        return ["lunch", "dinner"]
    if n == 3:
        return list(_BASE_SLOTS)
    slots = _BASE_SLOTS + ["snack"]
    slots += ["snack"] * max(0, n - 4)
    return slots


def active_slots(user_profile):
    slots = user_profile.get("slots")
    if slots:
        return [s for s in slots if s in {"breakfast", "lunch", "dinner", "snack"}]
    return derive_slots(user_profile.get("mealsPerDay"))
