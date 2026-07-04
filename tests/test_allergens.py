"""Allergen safety: every allergen id must actually exclude the meals that
contain its ingredients — no silent empty exclusion sets (handoff §8)."""

import vectorization as vz

# shellfish/pork have no ingredients in the 53-item vocabulary yet (handoff §8).
KNOWN_EMPTY = {"shellfish", "pork"}


def _meals_containing(catalog, ingredient_names):
    names = set(ingredient_names)
    return {i for i, rec in enumerate(catalog)
            if {c["ingredient"] for c in rec["composition"]} & names}


def test_allergen_map_ingredients_exist(bundle, ingredient_table):
    table = set(ingredient_table)
    for allergen, ings in bundle["allergen_map"].items():
        for ing in ings:
            assert ing in table, f"allergen {allergen!r} lists unknown ingredient {ing!r}"


def test_every_allergen_excludes_its_meals(bundle, catalog, ingredient_table):
    allergen_map = bundle["allergen_map"]
    full = vz.build_eligible_pool(catalog, ingredient_table, allergen_map)
    for allergen in allergen_map:
        filtered = vz.build_eligible_pool(catalog, ingredient_table, allergen_map, allergens=[allergen])
        removed = set(full) - set(filtered)
        expected = _meals_containing(catalog, allergen_map[allergen])
        assert removed == expected, f"{allergen}: removed {removed} != expected {expected}"
        if allergen not in KNOWN_EMPTY:
            assert removed, f"allergen {allergen!r} excluded nothing (silent-empty bug)"


def test_unknown_allergen_rejected(bundle):
    import profile_mapping as pm
    import pytest
    with pytest.raises(ValueError):
        pm.validate_allergens(["not_an_allergen"], bundle["allergen_map"])


def test_no_eligible_meal_contains_allergen(bundle, catalog, ingredient_table):
    allergen_map = bundle["allergen_map"]
    excluded = vz.get_excluded_ingredients(["dairy", "nuts"], allergen_map)
    eligible = vz.build_eligible_pool(catalog, ingredient_table, allergen_map, allergens=["dairy", "nuts"])
    for idx in eligible:
        ings = {c["ingredient"] for c in catalog[idx]["composition"]}
        assert not (ings & excluded)
