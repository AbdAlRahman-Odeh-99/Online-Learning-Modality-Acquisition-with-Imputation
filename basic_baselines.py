import numpy as np
from utils import match_cluster_labels
from learning import update_centers
from predict import predict_with_observed_views

# BASELINE: Always observe modality 1, never observe 2 or 3, no imputation, predict with observed view only.
def online_learning_first_view_only(X, Y, learned_centers, true_means, base_mask = [True, False, False]):
    """
    Online learning and prediction using only the first modality initially.
    
    Returns:
        total_reward: list of 0-1 rewards
        learned_centers: updated centers
        counts: counts per cluster-modality
    """
    K, M = learned_centers.shape
    counts = np.zeros((K, M))
    total_reward = []    
    for sample in range(len(X)):
        label_map = match_cluster_labels(learned_centers, true_means)
        observed_mask = base_mask.copy()
        final_pred = predict_with_observed_views(X[sample], learned_centers, observed_mask)
        matched_final_pred = label_map[final_pred]
        total_reward.append(int(matched_final_pred == Y[sample]))
        learned_centers, counts = update_centers(X[sample], Y[sample], learned_centers, counts, observed_mask)
    return total_reward, learned_centers, counts

# BASELINE: Always observe modality 1 and 2, never observe 3, no imputation, predict with observed view only.
def online_learning_first_second_views(X, Y, learned_centers, true_means, base_mask = [True, True, False]):
    """
    Online learning and prediction using only the first modality initially.
    
    Returns:
        total_reward: list of 0-1 rewards
        learned_centers: updated centers
        counts: counts per cluster-modality
    """
    K, M = learned_centers.shape
    counts = np.zeros((K, M))
    total_reward = []    
    for sample in range(len(X)):
        label_map = match_cluster_labels(learned_centers, true_means)
        observed_mask = base_mask.copy()
        final_pred = predict_with_observed_views(X[sample], learned_centers, observed_mask)
        matched_final_pred = label_map[final_pred]
        total_reward.append(int(matched_final_pred == Y[sample]))
        learned_centers, counts = update_centers(X[sample], Y[sample], learned_centers, counts, observed_mask)
    return total_reward, learned_centers, counts

# BASELINE: Always observe modality 1 and 3, never observe 2, no imputation, predict with observed view only.
def online_learning_first_third_views(X, Y, learned_centers, true_means, base_mask=[True, False, True]):
    """
    Online learning and prediction using only the first modality initially.
    
    Returns:
        total_reward: list of 0-1 rewards
        learned_centers: updated centers
        counts: counts per cluster-modality
    """
    K, M = learned_centers.shape
    counts = np.zeros((K, M))
    total_reward = []    
    for sample in range(len(X)):
        label_map = match_cluster_labels(learned_centers, true_means)
        observed_mask = base_mask.copy()
        final_pred = predict_with_observed_views(X[sample], learned_centers, observed_mask)
        matched_final_pred = label_map[final_pred]
        total_reward.append(int(matched_final_pred == Y[sample]))
        learned_centers, counts = update_centers(X[sample], Y[sample], learned_centers, counts, observed_mask)
    return total_reward, learned_centers, counts

# BASELINE: Always observe all modalities, no imputation needed, predict with all views.
def online_learning_all_views(X, Y, learned_centers, true_means):
    """
    Online learning and prediction using all views observed (no missing modalities).
    
    Returns:
        total_reward: list of 0-1 rewards
        learned_centers: updated centers
        counts: counts per cluster-modality
    """
    K, M = learned_centers.shape
    counts = np.zeros((K, M))
    total_reward = []
    observed_mask = np.ones(M, dtype=bool)  # all modalities observed
    for sample in range(len(X)):
        label_map = match_cluster_labels(learned_centers, true_means)
        final_pred = predict_with_observed_views(X[sample], learned_centers, observed_mask)
        matched_final_pred = label_map[final_pred]
        total_reward.append(int(matched_final_pred == Y[sample]))
        learned_centers, counts = update_centers(X[sample], Y[sample], learned_centers, counts, observed_mask)
    return total_reward, learned_centers, counts