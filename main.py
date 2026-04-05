import sys
import time
from pathlib import Path
import numpy as np
from utils import generate_view_combinations, generate_combination_costs, print_true_learned_means, combo_to_mask, match_cluster_labels
from data_generation import generate_synthetic_data
from learning import update_centers
from predict import predict_with_observed_views
from basic_baselines import online_learning_first_view_only, online_learning_first_second_views, online_learning_first_third_views, online_learning_all_views
from imputation import uniform_imputations, vote_all_combinations, vote_with_responsibilities, choose_best_combination_binary
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
start = time.time()

N_SAMPLES = 5000
N_SAMPLES_INIT_PERCENT = 0
K_CLUSTERS = 2
M_MODALITIES = 3
P_Y = [0.5, 0.5]
RANDOM_SEED = 0
COST_PER_MODALITY = {'1':0, '2':1, '3':1}
BUDGET = N_SAMPLES * sum(COST_PER_MODALITY.values()) * 0.7
remaining_budget = BUDGET


# Setting up the experiment
print("Generating synthetic data...")
X, Y, true_means, true_sigmas, rng = generate_synthetic_data(n_samples=N_SAMPLES, k_clusters=K_CLUSTERS, m_modalities=M_MODALITIES, p_y=np.array(P_Y), random_seed=RANDOM_SEED)
print("Generating view combinations & costs...")
view_combinations = generate_view_combinations(M_MODALITIES)[1:] # skip (1,) since it's always observed
combination_costs = generate_combination_costs(view_combinations, COST_PER_MODALITY)
comb_counts = {combo:0 for combo in view_combinations}
# print("Initializing K-means centers randomly & Counts...")
# learned_centers = rng.normal(loc=0.0, scale=1.0, size=(K_CLUSTERS, M_MODALITIES))
# counts = np.zeros((K_CLUSTERS, M_MODALITIES))
print(f"Initializing centers using first {N_SAMPLES_INIT_PERCENT*100:.0f}% samples...")
N_SAMPLES_INIT = N_SAMPLES * N_SAMPLES_INIT_PERCENT
if N_SAMPLES_INIT_PERCENT > 0:
    init_X = X[:int(N_SAMPLES_INIT)]
    init_Y = Y[:int(N_SAMPLES_INIT)]
    learned_centers = np.zeros((K_CLUSTERS, M_MODALITIES))
    for k in range(K_CLUSTERS):
        learned_centers[k] = init_X[init_Y == k].mean(axis=0)
    counts = np.zeros((K_CLUSTERS, M_MODALITIES))
    for k in range(K_CLUSTERS):
        counts[k, :] = np.sum(init_Y == k)
else:
    learned_centers = rng.normal(loc=0.0, scale=1.0, size=(K_CLUSTERS, M_MODALITIES))
    counts = np.zeros((K_CLUSTERS, M_MODALITIES))

print("Initializing base mask (free observed modality)...")
base_mask = np.zeros(M_MODALITIES, dtype=bool)
base_mask[0] = True


# First-view learning
total_reward_first, learned_centers_first, counts_first = online_learning_first_view_only(X[int(N_SAMPLES_INIT):], Y[int(N_SAMPLES_INIT):], learned_centers.copy(), true_means)

# First-Second-view learning
total_reward_first_second, learned_centers_first_second, counts_first_second = online_learning_first_second_views(X[int(N_SAMPLES_INIT):], Y[int(N_SAMPLES_INIT):], learned_centers.copy(), true_means)

# First-Third-view learning
total_reward_first_third, learned_centers_first_third, counts_first_third = online_learning_first_third_views(X[int(N_SAMPLES_INIT):], Y[int(N_SAMPLES_INIT):], learned_centers.copy(), true_means)

# All-view learning
total_reward_all, learned_centers_all, counts_all = online_learning_all_views(X[int(N_SAMPLES_INIT):], Y[int(N_SAMPLES_INIT):], learned_centers.copy(), true_means)

print("="*50)
print(f"First-view only learning: AVG Reward: {np.mean(total_reward_first):.3f}")
print("="*50)
print(f"First-Second-view learning: AVG Reward: {np.mean(total_reward_first_second):.3f}")
print("="*50)
print(f"First-Third-view learning: AVG Reward: {np.mean(total_reward_first_third):.3f}")
print("="*50)
print(f"All-view learning: AVG Reward: {np.mean(total_reward_all):.3f}")
print("="*50)
print("Starting online learning with adaptive view selection...")
total_reward = []
total_cost = []
for sample in range(int(N_SAMPLES_INIT), len(X)):
    observed_mask = base_mask.copy()

    if remaining_budget > 0:

        label_map = match_cluster_labels(learned_centers, true_means)
        aligned_centers = np.zeros_like(learned_centers)
        for learned_k, true_k in label_map.items():
            aligned_centers[true_k] = learned_centers[learned_k]

        imputations, cluster_ids = uniform_imputations(
            x_sample=X[sample],
            centers=aligned_centers,
            observed_mask=observed_mask,
            n_samples_per_cluster=5,
            radius=0.5,
            rng=rng
        )

        final_preds, confs = vote_all_combinations(imputations, aligned_centers, view_combinations)
        # final_preds, confs = vote_with_responsibilities(
        #     x_sample=X[sample],
        #     observed_mask=observed_mask,
        #     imputations=imputations,
        #     cluster_ids=cluster_ids,
        #     centers=learned_centers,
        #     view_combinations=view_combinations
        # )
        #matched_preds = np.array([label_map[p] for p in final_preds])
        best_combo, best_score, scores = choose_best_combination_binary(
            view_combinations,
            #matched_preds,
            final_preds,
            Y[sample],
            confs,
            combination_costs,
            eta=0.2
        )
        if remaining_budget - combination_costs[best_combo] < 0:
            best_combo = (1,)  # fallback to first modality only
            total_cost.append(0)
        #elif not above_threshold:
        #    best_combo = (1,)  # fallback to first modality only
        #    total_cost.append(0)
        else:
            remaining_budget -= combination_costs[best_combo]
            total_cost.append(combination_costs[best_combo])
        observed_mask = combo_to_mask(best_combo, M_MODALITIES)

    comb_counts[best_combo] += 1
    final_pred = predict_with_observed_views(X[sample], aligned_centers, observed_mask) 
    #matched_final_pred = label_map[final_pred]
    #instance_reward = int(matched_final_pred == Y[sample])
    instance_reward = int(final_pred == Y[sample])
    total_reward.append(instance_reward)
    
    learned_centers, counts = update_centers(
        X[sample],
        Y[sample],
        #final_pred,
        learned_centers,
        counts,
        observed_mask
    )
    

#print_true_learned_means(M_MODALITIES, true_means, learned_centers)
print(f"Average reward over {len(total_reward)} samples: {np.mean(total_reward):.3f}, Total cost: {np.sum(total_cost)}")
print(f"Combination usage counts: {comb_counts}")
end = time.time()
print("Execution time:", end - start, "seconds")