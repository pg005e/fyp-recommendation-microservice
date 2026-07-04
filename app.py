"""Flask entry point: POST /generate-plan + GET /health (handoff §6, §11)."""

import logging

from flask import Flask, jsonify, request

from config import PORT
from recommender import MealRecommender

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# Load all artifacts once on startup (~seconds); /health gates traffic until ready.
recommender = MealRecommender()


@app.get("/health")
def health():
    h = recommender.health()
    return jsonify(h), (200 if h["status"] == "ready" else 503)


@app.post("/generate-plan")
def generate_plan():
    data = request.get_json(force=True, silent=True) or {}
    try:
        plan = recommender.generate(
            user_profile=data.get("userProfile"),
            adaptive_profile=data.get("adaptiveProfile"),
        )
    except ValueError as e:
        return jsonify({"error": "invalid_request", "detail": str(e)}), 400
    except Exception as e:  # noqa: BLE001 — surface unexpected failures, don't 200 a broken plan
        logging.exception("generate-plan failed")
        return jsonify({"error": "internal", "detail": str(e)}), 500
    return jsonify(plan)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
