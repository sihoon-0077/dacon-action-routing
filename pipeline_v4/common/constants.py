import numpy as np

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
LABEL2ID = {c: i for i, c in enumerate(ALL_CLASSES)}
ID2LABEL = {i: c for i, c in enumerate(ALL_CLASSES)}
NUM_CLASSES = 14

COARSE_NAMES = ["inspect", "modify", "execute", "communicate"]
COARSE_OF = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])

SEED = 42
N_FOLDS = 5
DATA_DIR = "./data"
FOLD_FILE = "./pipeline_v4/folds/fold_assignments.csv"

TRAIN_COUNTS_APPROX = {
    "edit_file": 11171, "grep_search": 9912, "read_file": 9257,
    "glob_pattern": 5284, "respond_only": 5178, "run_bash": 5068,
    "apply_patch": 4823, "run_tests": 4561, "list_directory": 4329,
    "ask_user": 2701, "plan_task": 2679, "lint_or_typecheck": 2283,
    "write_file": 1481, "web_search": 1273,
}


def session_of(sample_id: str) -> str:
    return sample_id.rsplit("-step_", 1)[0]
