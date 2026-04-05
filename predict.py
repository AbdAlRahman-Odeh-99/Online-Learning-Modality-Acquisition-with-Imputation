import numpy as np

def predict_with_observed_views(x, centers, observed_mask):
    observed_x = x[observed_mask]
    observed_centers = centers[:, observed_mask]
    distances = np.sum((observed_centers - observed_x) ** 2, axis=1)
    return np.argmin(distances)

def predict_all_combinations(x_sample, centers, view_combinations):
    """
    Predict clusters for all non-empty combinations of modalities for a fully observed sample.

    Args:
        x_sample: array (M,), fully imputed sample
        centers: array (K, M), learned cluster centers

    Returns:
        preds: array, predicted cluster for each combination
    """
    preds = []
    for combo in view_combinations:
        mask = np.zeros(len(x_sample), dtype=bool)
        mask[np.array(combo) - 1] = True
        pred = predict_with_observed_views(x_sample, centers, mask)
        preds.append(pred)
    return np.array(preds)
