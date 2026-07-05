"""
Deterministic 7-day plan assembly (handoff §9). Pure arithmetic + selection
rules over the per-slot ranked candidates — no model, no randomness.

Steps: (1) per-slot candidates in, (2) optional-slot drop via slotAdherence,
(3) per-day top pick with no-consecutive-day repeat, (4) ±tolerance portion
scaling, (5) variety cap of <= max_repeats per week.
"""

SLOT_ADHERENCE_DROP = 0.2  # drop a slot the user fills less than this often


def _pick(candidates, usage, max_repeats, avoid):
    """Top-ranked candidate honoring variety cap + no-consecutive-repeat, with
    graceful relaxation if the pool is exhausted."""
    for c in candidates:                       # ideal: fresh + not yesterday's
        if usage.get(c, 0) < max_repeats and c != avoid:
            return c
    for c in candidates:                       # relax the consecutive rule
        if usage.get(c, 0) < max_repeats:
            return c
    for c in candidates:                        # relax the variety cap
        if c != avoid:
            return c
    return candidates[0] if candidates else None


def _scaled_recipe(slot, meal, scale):
    ingredients = [
        {
            "foodName": ing["foodName"],
            "grams": round(ing["grams"] * scale, 1),
            "calories": round(ing["calories"] * scale, 1),
            "protein": round(ing["protein"] * scale, 2),
            "carbs": round(ing["carbs"] * scale, 2),
            "fat": round(ing["fat"] * scale, 2),
        }
        for ing in meal["ingredients"]
    ]
    return {
        "recipeId": meal["recipeId"],
        "mealType": slot,
        "recipeName": meal["name"],
        "prepNotes": None,
        "calories": round(meal["calories"] * scale, 1),
        "protein": round(meal["protein"] * scale, 2),
        "carbs": round(meal["carbs"] * scale, 2),
        "fat": round(meal["fat"] * scale, 2),
        "ingredients": ingredients,
    }


def assemble_plan(slots, ranked_by_slot, meal_info, target_calories,
                  slot_adherence=None, calorie_tolerance=0.10,
                  max_repeats=2, days=7):
    slot_adherence = slot_adherence or {}

    # Honor the user's full slot structure. (Previously we dropped slots with
    # slotAdherence < 0.2, but a single partially-logged week pushes most slots
    # below that — unlogged days count as 0 — which collapsed the plan to ~1
    # meal/day. Slot-softening needs stronger, multi-week evidence, so it's
    # disabled here rather than punishing sparse logging.)
    positions = list(slots)

    usage = {}
    last_by_position = {}
    plan_days = []

    for day in range(1, days + 1):
        chosen = []  # (position_index, slot, meal_id)
        for pos, slot in enumerate(positions):
            candidates = ranked_by_slot.get(slot, [])
            meal_id = _pick(candidates, usage, max_repeats, last_by_position.get(pos))
            if meal_id is None:
                continue
            usage[meal_id] = usage.get(meal_id, 0) + 1
            last_by_position[pos] = meal_id
            chosen.append((slot, meal_id))

        # Step 4: scale the whole day toward the calorie target if off by > tol.
        day_calories = sum(meal_info[c]["calories"] for _, c in chosen)
        scale = 1.0
        if day_calories > 0 and abs(day_calories - target_calories) / target_calories > calorie_tolerance:
            scale = target_calories / day_calories

        recipes = [_scaled_recipe(slot, meal_info[c], scale) for slot, c in chosen]
        plan_days.append({
            "day": day,
            "recipes": recipes,
            "calories": round(sum(r["calories"] for r in recipes), 1),
            "protein": round(sum(r["protein"] for r in recipes), 2),
            "carbs": round(sum(r["carbs"] for r in recipes), 2),
            "fat": round(sum(r["fat"] for r in recipes), 2),
        })

    return plan_days
