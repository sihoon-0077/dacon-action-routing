from collections import Counter, defaultdict

import numpy as np

from src.constants import ACTIONS
from src.state_features import get_last_actions


def build_last2_prior(samples, smooth=1.0):
    counts = defaultdict(Counter)
    global_counts = Counter()
    for sample in samples:
        if "action" not in sample:
            continue
        actions = get_last_actions(sample, 2)
        key = "|".join(actions) if actions else "NONE"
        counts[key][sample["action"]] += 1
        global_counts[sample["action"]] += 1
    global_total = sum(global_counts.values())
    global_row = np.array([(global_counts[a] + smooth) / (global_total + smooth * len(ACTIONS)) for a in ACTIONS])
    prior = {}
    for key, counter in counts.items():
        total = sum(counter.values())
        prior[key] = np.array([(counter[a] + smooth) / (total + smooth * len(ACTIONS)) for a in ACTIONS])
    return prior, global_row


def prior_matrix(samples, prior, global_row):
    rows = []
    for sample in samples:
        actions = get_last_actions(sample, 2)
        key = "|".join(actions) if actions else "NONE"
        rows.append(prior.get(key, global_row))
    return np.vstack(rows)
