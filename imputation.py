import numpy as np
from collections import Counter
from utils import posterior_probability
from predict import predict_all_combinations_proba



def compute_responsibilities(x_sample, centers, observed_mask):
    """
    Compute soft cluster assignments (responsibilities) given only the observed modalities.
    
    x_sample:      shape (M,)
    centers:       shape (K, M)
    observed_mask: shape (M,) boolean
    
    Returns: responsibilities shape (K,) summing to 1
    """
    K = centers.shape[0]
    log_probs = np.zeros(K)

    for k in range(K):
        # Squared distance between x_sample and cluster center, observed modalities only
        diff = x_sample[observed_mask] - centers[k, observed_mask]
        log_probs[k] = -0.5 * np.dot(diff, diff)  # log of Gaussian likelihood (ignoring constants)

    # Softmax for numerical stability
    log_probs -= np.max(log_probs)
    probs = np.exp(log_probs)
    responsibilities = probs / probs.sum()

    return responsibilities  # shape (K,)

def compute_scores(view_combinations, total_instances, cluster, cluster_modality_counts, combo_costs, rho, eta=0.2, c_param=1.41):
    scores = []
    
    for i, combo in enumerate(view_combinations):
        # 1. Exploitation: Expected Reward
        r = rho[i]
        expected_reward = (r**2) + (1 - r)**2

        # 2. Exploration: UCB Bonus
        combo_indices = [m - 1 for m in combo]  # 1-indexed to 0-indexed
        combo_obs_count = cluster_modality_counts[cluster, combo_indices].min()

        if total_instances == 0 or combo_obs_count == 0:
            eps = 1e6
        else:
            eps = c_param * np.sqrt(np.log(total_instances) / combo_obs_count)

        # 3. Cost penalty
        cost_penalty = eta * combo_costs[combo]

        # 4. Total Utility Score
        score = expected_reward + eps - cost_penalty
        scores.append(score)
    
    return np.array(scores)

def majority_vote(decisions, rng):
    counts = Counter(decisions)
    max_count = max(counts.values())
    tied_combos = [combo for combo, count in counts.items() if count == max_count]
    return tied_combos[rng.choice(len(tied_combos))]

def oneshot_acquisition(
        view_combinations,
        x_sample,
        centers,
        observed_mask,
        total_instances,
        combo_costs,
        cluster_modality_counts,
        eta=0.2,
        c_param=1.41,
        rng=None
    ):

    if rng is None:
        rng = np.random.default_rng()

    M = x_sample.shape[0]
    K = centers.shape[0]
    missing_mask = ~observed_mask
    all_scores = []  # will be shape (K * n_samples_per_cluster, n_combos)
    max_decisions = []  # to track which combo was best for each imputation

    # Compute responsibilities q_{y_k | x_{0,t}} from the free view
    responsibilities = compute_responsibilities(x_sample, centers, observed_mask)  # shape (K,)
    #print(f"Responsibilities: {responsibilities}")

    for k in range(K):    
        x_completed = x_sample.copy()
        # Imputation: Use the conditional mean of the missing modalities given the observed ones, which is just the cluster center's values for those modalities.
        x_completed[missing_mask] = centers[k, missing_mask]
        # Prediction for each combo
        soft_preds = predict_all_combinations_proba(x_completed, centers, view_combinations) # shape (n_combos,)
        #print(f"Soft predictions for cluster {k}: {soft_preds}")
        # rho per combo for this specific imputation, weighted by cluster responsibility
        rho = responsibilities[k] * soft_preds # shape (n_combos,)
        #print(f"Rho values for cluster {k}: {rho}")
        
        # score per combo for this specific imputation
        scores = compute_scores(
            view_combinations=view_combinations,
            total_instances=total_instances,
            cluster=k,
            cluster_modality_counts=cluster_modality_counts,
            combo_costs=combo_costs,
            rho=rho,
            eta=eta,
            c_param=c_param
        )  # shape (n_combos,)

        all_scores.append(scores)

    all_scores = np.array(all_scores)  # shape (K * n_samples_per_cluster, n_combos)
    #print(f"All scores shape: {all_scores.shape}")
    #print(f"All scores: {all_scores}")
    
    max_decisions = []
    for row_scores in all_scores:
        max_score = np.max(row_scores)
        tied_indices = np.where(row_scores == max_score)[0]
        best_idx = rng.choice(tied_indices)  # random tie-break
        max_decisions.append(view_combinations[best_idx])

    #print(f"Best combos for each imputation: {max_decisions}")

    majority_vote_decision = majority_vote(max_decisions, rng)
    
    #print(f"Majority vote decision: {majority_vote_decision}")
    
    return majority_vote_decision