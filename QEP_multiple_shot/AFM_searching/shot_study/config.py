"""Single source of truth for production and smoke-test experiment grids."""
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = STUDY_DIR / "data" / "shots_2026-09-11"
FIG_DIR = STUDY_DIR / "figures" / "shots_2026-09-11"
LOG_DIR = STUDY_DIR / "logs"

N_LIST = (5, 9)
BETA = 0.1
LEARNING_RATE = 0.01
N_Y = 100
TARGET = 1.0
T = 100.0
DT = 0.1

# Fixed_TN uses [10,20,50,100,200,500,1000,2000,5000].  Here N_g=10000 is
# added because task 2 asks for it explicitly.
NG_GRID = (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
TRAJECTORY_NG = (10, 100, 1000, 10000)
BETA_GRID = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)

# Drawn once with numpy.default_rng(20260911), then frozen here so every job
# and future rerun uses exactly the same "randomly chosen" starting points.
INITIAL_POINTS = (
    (1.0285259364272148, 2.116181487375421),
    (0.6954632245521951, 1.9556663479392087),
    (0.8874683777374885, 1.6020526680779208),
)
TYPICAL_POINT = (0.5, 1.5)

GRADIENT_REPEATS = 256
TRAJECTORY_SEEDS = 5
TRAJECTORY_EPOCHS = 200
STATS_EPOCHS = 500
STATS_NG = 1000

# Same numerical budgets as Fixed_TN.  Here they count gradient shots, so a
# run with N_g shots in each of two noncommuting measurement groups receives
# floor(budget/(2*N_g)) updates.
SHOT_BUDGETS = (5000, 10000, 20000, 50000, 100000)
BUDGET_SEEDS = 5


def ensure_dirs():
    for path in (DATA_DIR, FIG_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)
