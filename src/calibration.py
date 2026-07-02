import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import log_loss


def temperature_scale_logits(logits, temperature):
    return np.asarray(logits, dtype=np.float64) / float(temperature)


def fit_temperature(logits, y_true_ids, labels=None, bounds=(0.2, 5.0)):
    logits = np.asarray(logits, dtype=np.float64)
    y_true_ids = np.asarray(y_true_ids, dtype=np.int64)
    labels = list(range(logits.shape[1])) if labels is None else labels

    def objective(temp):
        probs = softmax(temperature_scale_logits(logits, temp), axis=1)
        return log_loss(y_true_ids, probs, labels=labels)

    result = minimize_scalar(objective, bounds=bounds, method="bounded", options={"xatol": 1e-4})
    temp = float(result.x)
    probs = softmax(temperature_scale_logits(logits, temp), axis=1)
    return temp, probs, float(result.fun)


def logits_to_probs(logits):
    logits = np.asarray(logits, dtype=np.float64)
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
