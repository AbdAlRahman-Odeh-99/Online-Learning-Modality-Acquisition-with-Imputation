import numpy as np

# This function is used to HARD predict the cluster for a sample with only some modalities observed.
def predict_with_observed_views(x, centers, observed_mask):
    observed_x = x[observed_mask]
    observed_centers = centers[:, observed_mask]
    distances = np.sum((observed_centers - observed_x) ** 2, axis=1)
    return np.argmin(distances)

# This function is used to SOFT predict the cluster probabilities for a sample with only some modalities observed.
def predict_proba_with_observed_views(x, centers, observed_mask):
    """Soft probability of each cluster given observed modalities."""
    observed_x = x[observed_mask]
    observed_centers = centers[:, observed_mask]
    dists = np.sum((observed_centers - observed_x) ** 2, axis=1)  # shape (K,)
    log_probs = -0.5 * dists
    log_probs -= np.max(log_probs)  # numerical stability
    probs = np.exp(log_probs) / np.exp(log_probs).sum()  # softmax, shape (K,)
    return probs  # probability per cluster

# This function is used to SOFT predict the cluster probabilities for each combo of observed modalities for a fully observed sample.
def predict_all_combinations_proba(x_sample, centers, view_combinations):
    """
    Returns soft probability for the positive class (cluster 1) for each combo. Shape (n_combos,)
    """
    probs = []
    for combo in view_combinations:
        mask = np.zeros(len(x_sample), dtype=bool)
        mask[np.array(combo) - 1] = True
        cluster_probs = predict_proba_with_observed_views(x_sample, centers, mask)  # shape (K,)
        probs.append(cluster_probs[1])
    return np.array(probs)  # shape (n_combos,)









# This function is used to HARD predict the cluster for each combo of observed modalities for a fully observed sample.
# def predict_all_combinations(x_sample, centers, view_combinations):
#     """
#     Predict clusters for all non-empty combinations of modalities for a fully observed sample.

#     Args:
#         x_sample: array (M,), fully imputed sample
#         centers: array (K, M), learned cluster centers

#     Returns:
#         preds: array, predicted cluster for each combination
#     """
#     preds = []
#     for combo in view_combinations:
#         mask = np.zeros(len(x_sample), dtype=bool)
#         mask[np.array(combo) - 1] = True
#         pred = predict_with_observed_views(x_sample, centers, mask)
#         preds.append(pred)
#     return np.array(preds)