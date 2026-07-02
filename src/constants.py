ACTIONS = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
    "edit_file",
    "write_file",
    "apply_patch",
    "run_bash",
    "run_tests",
    "lint_or_typecheck",
    "ask_user",
    "plan_task",
    "web_search",
    "respond_only",
]

ALL_CLASSES = ACTIONS
ACTION_TO_ID = {action: i for i, action in enumerate(ACTIONS)}
LABEL2ID = ACTION_TO_ID
ID_TO_ACTION = {i: action for action, i in ACTION_TO_ID.items()}

COARSE_NAMES = ["inspect", "modify", "execute", "communicate"]
COARSE_OF = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
SEED = 42
VAL_RATIO = 0.2

ACTION_TO_GROUP4 = {
    "read_file": "inspect",
    "grep_search": "inspect",
    "list_directory": "inspect",
    "glob_pattern": "inspect",
    "edit_file": "modify",
    "write_file": "modify",
    "apply_patch": "modify",
    "run_bash": "execute",
    "run_tests": "execute",
    "lint_or_typecheck": "execute",
    "ask_user": "communicate",
    "plan_task": "communicate",
    "web_search": "communicate",
    "respond_only": "communicate",
}

GROUP4_TO_ACTIONS = {
    "inspect": ["read_file", "grep_search", "list_directory", "glob_pattern"],
    "modify": ["edit_file", "write_file", "apply_patch"],
    "execute": ["run_bash", "run_tests", "lint_or_typecheck"],
    "communicate": ["ask_user", "plan_task", "web_search", "respond_only"],
}
