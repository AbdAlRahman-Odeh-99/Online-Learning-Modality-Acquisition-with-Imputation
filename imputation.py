import numpy as np
from collections import Counter
from utils import posterior_probability
from predict import predict_all_combinations


def uniform_imputations(x_sample, centers, observed_mask, n_samples_per_cluster=5, radius=0.5, rng=None):
    """
    Impute missing views with uniform samples around cluster centers.

    Parameters
    ----------
    x_sample : array (M,)
    centers : array (K, M)
    observed_mask : bool (M,)
    n_samples_per_cluster : int, number of samples per cluster
    radius : float, half-width of uniform interval around center
    rng : np.random.Generator, optional

    Returns
    -------
    imputations : array (K * n_samples_per_cluster, M)
        Each row is one candidate imputed sample
    cluster_ids : array (K * n_samples_per_cluster,)
        Cluster index corresponding to each imputation
    """
    if rng is None:
        rng = np.random.default_rng()

    M = x_sample.shape[0]
    K = centers.shape[0]
    missing_mask = ~observed_mask # Flip to get missing modalities

    imputations = []
    cluster_ids = []

    for k in range(K): # Loop over clusters
        for _ in range(n_samples_per_cluster): # Loop to create multiple imputations per cluster
            x_completed = x_sample.copy()
            # Uniform sample for missing modalities
            x_completed[missing_mask] = rng.uniform(low=centers[k, missing_mask] - radius, high=centers[k, missing_mask] + radius)
            imputations.append(x_completed)
            cluster_ids.append(k)

    return np.array(imputations), np.array(cluster_ids)

def majority_vote(predictions):
    """
    Majority vote over candidate predictions.

    Args:
        predictions: array-like

    Returns:
        final_prediction
        confidence
    """
    counts = Counter(predictions)

    final_prediction = counts.most_common(1)[0][0]
    confidence = counts[final_prediction] / len(predictions)

    return final_prediction, confidence

def vote_all_combinations(imputations, centers, view_combinations):
    """
    Predict all combinations for all imputations,
    then majority vote per combination.

    Returns:
        final_preds : one prediction per combination
        confidences : one confidence per combination
    """
    all_preds = []

    for imp in imputations:
        preds = predict_all_combinations(imp, centers, view_combinations)
        all_preds.append(preds)

    all_preds = np.array(all_preds)   # shape = (n_imputations, n_combinations)

    final_preds = []
    confidences = []

    for j in range(all_preds.shape[1]):
        pred, conf = majority_vote(all_preds[:, j])
        final_preds.append(pred)
        confidences.append(conf)

    return np.array(final_preds), np.array(confidences)

def choose_best_combination_binary(view_combinations, final_preds, true_label, confs, combination_costs, eta, epsilons=None, threshold = 0):
    """
    Choose best combination using binary reward.

    Args:
        final_preds : predicted label for each combination
        true_label : true sample label
    """

    scores = []

    for combo, pred in zip(view_combinations, final_preds):

        reward = int(pred == true_label)
        cost = combination_costs[combo]

        if epsilons is None:
            eps = 0.0
        else:
            eps = epsilons.get(combo, 0.0)

        score = reward + eps - eta * cost
        scores.append(score)

    #print(f"Combination scores: {dict(zip(view_combinations, scores))}")
    #print(f"Combination confidences: {dict(zip(view_combinations, confs))}")
    # Step 1: find maximum score
    max_score = np.max(scores)

    # Step 2: all candidates with same max score
    candidate_idxs = np.where(scores == max_score)[0]

    # Step 3: tie-break using confidence
    if len(candidate_idxs) > 1:
        best_idx = candidate_idxs[np.argmax(confs[candidate_idxs])]
    else:
        best_idx = candidate_idxs[0]

    #print(f"Chosen combination: {view_combinations[best_idx]}, Score: {scores[best_idx]:.3f}, Confidence: {confs[best_idx]:.3f}")
    return view_combinations[best_idx], scores[best_idx], np.array(scores)

def vote_with_responsibilities(x_sample, observed_mask, imputations, cluster_ids, centers, view_combinations):
    """
    imputations: (S*K, M) - all generated candidates
    cluster_ids: (S*K,) - which cluster generated each candidate
    """
    # Step A: Get weights for each cluster based on original observed data
    alphas = posterior_probability(x_sample, centers, observed_mask)
    
    all_preds = []
    # Step B: Get predictions for all candidates and all combinations
    for imp in imputations:
        preds = predict_all_combinations(imp, centers, view_combinations)
        all_preds.append(preds)
    
    all_preds = np.array(all_preds) # (n_imputations, n_combinations)
    K = centers.shape[0]
    
    final_preds = []
    confidences = []

    # Step C: Weighted Voting per Combination
    for j in range(all_preds.shape[1]):
        weighted_counts = np.zeros(K)
        
        for s in range(len(all_preds)):
            pred_label = all_preds[s, j]
            parent_cluster = cluster_ids[s]
            
            # Instead of +1, we add the responsibility weight
            weighted_counts[pred_label] += alphas[parent_cluster]
            
        best_pred = np.argmax(weighted_counts)
        conf = weighted_counts[best_pred] / np.sum(weighted_counts)
        
        final_preds.append(best_pred)
        confidences.append(conf)
    
    return np.array(final_preds), np.array(confidences)
