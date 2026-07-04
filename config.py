import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Artifacts live where the user uploaded them (track-treat/artifacts); override
# with ARTIFACTS_DIR for Docker / alternate layouts.
ARTIFACTS_DIR = os.environ.get(
    "ARTIFACTS_DIR",
    os.path.abspath(os.path.join(BASE_DIR, "..", "track-treat", "artifacts")),
)

PORT = int(os.environ.get("PORT", "5001"))

# Cold-start / fallback knobs.
DEFAULT_COMPLEXITY_MAX = 10          # no restriction until an adaptive profile exists
PREFERRED_BLEND_WEIGHT = 0.30        # weight of preferred-recipe centroid in the user vector
CALORIE_TOLERANCE = 0.10             # ±10% before portion scaling kicks in
MAX_REPEATS_PER_WEEK = 2             # a meal may appear at most twice across 7 days
