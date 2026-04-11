import sys
import time
from pathlib import Path
import numpy as np
from utils import generate_view_combinations, generate_combination_costs, print_true_learned_means, combo_to_mask, match_cluster_labels, plot_combo_selection
from data_generation import generate_synthetic_data
from learning import update_centers
from predict import predict_with_observed_views
from basic_baselines import online_learning_first_view_only, online_learning_first_second_views, online_learning_first_third_views, online_learning_all_views
from imputation import oneshot_acquisition, sequential_acquisition
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
start = time.time()

N_SAMPLES = 5000
N_SAMPLES_INIT_PERCENT = 0
K_CLUSTERS = 2
M_MODALITIES = 3
P_Y = [0.5, 0.5]
#RANDOM_SEED = [0, 10, 20, 30, 42]
RANDOM_SEED = [0]
COST_PER_MODALITY = {'1':0, '2':1, '3':1}
BUDGET = N_SAMPLES * sum(COST_PER_MODALITY.values()) * 0.7
C_PARAM = 1.41  # Exploration constant (sqrt(2) is standard)
STOPPING_THRESHOLD = 0.6

# For storing results across trials
trial_first_view_only = []
trial_first_second_views = []
trial_first_third_views = []
trial_all_views = []
trial_oneshot_reward = []
trial_oneshot_cost = []
trial_seq_reward = []
trial_seq_cost = []

for seed in RANDOM_SEED:
    print(f"\n=== Starting trial with random seed {seed} ===")
    # Setting up the experiment
    oneshot_combo_selection_history = []
    seq_combo_selection_history = []
    print("Generating synthetic data...")
    X, Y, true_means, true_sigmas, rng = generate_synthetic_data(n_samples=N_SAMPLES, k_clusters=K_CLUSTERS, m_modalities=M_MODALITIES, p_y=np.array(P_Y), random_seed=seed)
    print("Generating view combinations & costs...")
    view_combinations = generate_view_combinations(M_MODALITIES)[1:] # skip (1,) since it's always observed
    combo_costs = generate_combination_costs(view_combinations, COST_PER_MODALITY)
    
    print(f"Initializing centers using first {N_SAMPLES_INIT_PERCENT*100:.0f}% samples...")
    N_SAMPLES_INIT = N_SAMPLES * N_SAMPLES_INIT_PERCENT
    if N_SAMPLES_INIT_PERCENT > 0:
        init_X = X[:int(N_SAMPLES_INIT)]
        init_Y = Y[:int(N_SAMPLES_INIT)]
        learned_centers = np.zeros((K_CLUSTERS, M_MODALITIES))
        for k in range(K_CLUSTERS):
            learned_centers[k] = init_X[init_Y == k].mean(axis=0)
        cluster_modality_counts  = np.zeros((K_CLUSTERS, M_MODALITIES))
        for k in range(K_CLUSTERS):
            cluster_modality_counts [k, :] = np.sum(init_Y == k)
    else:
        learned_centers = rng.normal(loc=0.0, scale=1.0, size=(K_CLUSTERS, M_MODALITIES))
        cluster_modality_counts = np.zeros((K_CLUSTERS, M_MODALITIES))

    print("Initializing base mask (free observed modality)...")
    base_mask = np.zeros(M_MODALITIES, dtype=bool)
    base_mask[0] = True

    '''
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
    trial_first_view_only.append(np.mean(total_reward_first))
    print("="*50)
    print(f"First-Second-view learning: AVG Reward: {np.mean(total_reward_first_second):.3f}")
    trial_first_second_views.append(np.mean(total_reward_first_second))
    print("="*50)
    print(f"First-Third-view learning: AVG Reward: {np.mean(total_reward_first_third):.3f}")
    trial_first_third_views.append(np.mean(total_reward_first_third))
    print("="*50)
    print(f"All-view learning: AVG Reward: {np.mean(total_reward_all):.3f}")
    trial_all_views.append(np.mean(total_reward_all))
    print("="*50)
    '''
    
    print("Starting online learning with adaptive view selection (ONESHOT)...")
    total_instances = 0
    remaining_budget = BUDGET
    oneshot_total_reward = []
    oneshot_total_cost = []
    oneshot_cluster_modality_counts = cluster_modality_counts.copy()
    oneshot_learned_centers = learned_centers.copy()
    for sample in range(int(N_SAMPLES_INIT), len(X)):
        total_instances += 1
        observed_mask = base_mask.copy()

        if remaining_budget > 0:
            label_map = match_cluster_labels(oneshot_learned_centers, true_means)
            
            best_combo = oneshot_acquisition(
                view_combinations=view_combinations,
                x_sample=X[sample],
                centers=oneshot_learned_centers,
                observed_mask=observed_mask,
                total_instances=total_instances,
                combo_costs=combo_costs,
                cluster_modality_counts=oneshot_cluster_modality_counts,
                eta=0.2,
                c_param=C_PARAM,
                rng=rng
            )

            if remaining_budget - combo_costs[best_combo] < 0:
                best_combo = (1,)  # fallback to first modality only
                oneshot_total_cost.append(0)
            #elif not above_threshold:
            #    best_combo = (1,)  # fallback to first modality only
            #    total_cost.append(0)
            else:
                remaining_budget -= combo_costs[best_combo]
                oneshot_total_cost.append(combo_costs[best_combo])
            observed_mask = combo_to_mask(best_combo, M_MODALITIES)

        final_pred = predict_with_observed_views(X[sample], oneshot_learned_centers, observed_mask) 
        matched_final_pred = label_map[final_pred]
        instance_reward = int(matched_final_pred == Y[sample])
        oneshot_total_reward.append(instance_reward)
        #observed_indices = np.where(observed_mask)[0]
        oneshot_combo_selection_history.append(best_combo)

        oneshot_learned_centers, oneshot_cluster_modality_counts  = update_centers(
            X[sample],
            Y[sample],
            oneshot_learned_centers,
            oneshot_cluster_modality_counts,
            observed_mask
        )

    #print_true_learned_means(M_MODALITIES, true_means, oneshot_learned_centers)
    print(oneshot_cluster_modality_counts)
    print(f"Average reward over {len(oneshot_total_reward)} samples: {np.mean(oneshot_total_reward):.3f}, Total cost: {np.sum(oneshot_total_cost)}")
    print("Combo selection counts:")
    for combo in view_combinations:
        count = sum(1 for c in oneshot_combo_selection_history if c == combo)
        print(f"  {combo}: {count}")
    plot_combo_selection(oneshot_combo_selection_history, view_combinations, title=f"Oneshot Combo Selection History (Seed {seed})", save_path=f"plots/oneshot_combo_selection_history_seed_{seed}.png")
    trial_oneshot_reward.append(np.mean(oneshot_total_reward))
    trial_oneshot_cost.append(np.sum(oneshot_total_cost))

    print("="*50)

    
    # SEQUENTIAL ACQUISITION
    print("Starting online learning with adaptive view selection (SEQUENTIAL)...")
    total_instances = 0
    remaining_budget = BUDGET
    seq_total_reward = []
    seq_total_cost = []
    seq_cluster_modality_counts = cluster_modality_counts.copy()
    seq_learned_centers = learned_centers.copy()
    for sample in range(int(N_SAMPLES_INIT), len(X)):
        total_instances += 1
        observed_mask = base_mask.copy()

        if remaining_budget > 0:
            label_map = match_cluster_labels(seq_learned_centers, true_means)
            
            best_combo, remaining_budget = sequential_acquisition(
                view_combinations=view_combinations,
                x_sample=X[sample],
                centers=seq_learned_centers,
                observed_mask=observed_mask,
                total_instances=total_instances,
                combo_costs=combo_costs,
                cluster_modality_counts=seq_cluster_modality_counts,
                remaining_budget=remaining_budget,
                eta=0.2,
                c_param=C_PARAM,
                stopping_threshold=STOPPING_THRESHOLD,
                rng=rng
            )

            if best_combo == (1,):
                seq_total_cost.append(0)    
            else:
                seq_total_cost.append(combo_costs[best_combo])
            observed_mask = combo_to_mask(best_combo, M_MODALITIES)

        final_pred = predict_with_observed_views(X[sample], seq_learned_centers, observed_mask) 
        matched_final_pred = label_map[final_pred]
        instance_reward = int(matched_final_pred == Y[sample])
        seq_total_reward.append(instance_reward)
        #observed_indices = np.where(observed_mask)[0]
        seq_combo_selection_history.append(best_combo)

        seq_learned_centers, seq_cluster_modality_counts  = update_centers(
            X[sample],
            Y[sample],
            seq_learned_centers,
            seq_cluster_modality_counts,
            observed_mask
        )

        #if sample == 0:
        #    break

    print(seq_cluster_modality_counts)
    print(f"Average reward over {len(seq_total_reward)} samples: {np.mean(seq_total_reward):.3f}, Total cost: {np.sum(seq_total_cost)}")
    print("Combo selection counts:")
    for combo in view_combinations:
        count = sum(1 for c in seq_combo_selection_history if c == combo)
        print(f"  {combo}: {count}")
    plot_combo_selection(seq_combo_selection_history, view_combinations, title=f"Combo Selection History (Seed {seed})", save_path=f"plots/combo_selection_history_seed_{seed}.png")
    trial_seq_reward.append(np.mean(seq_total_reward))
    trial_seq_cost.append(np.sum(seq_total_cost))

print("\nAll trials completed.")
#print(f"Average reward (First View Only): {np.mean(trial_first_view_only):.3f} ± {np.std(trial_first_view_only):.3f}")
#print(f"Average reward (First-Second Views): {np.mean(trial_first_second_views):.3f} ± {np.std(trial_first_second_views):.3f}")
#print(f"Average reward (First-Third Views): {np.mean(trial_first_third_views):.3f} ± {np.std(trial_first_third_views):.3f}")
#print(f"Average reward (All Views): {np.mean(trial_all_views):.3f} ± {np.std(trial_all_views):.3f}")
print(f"Average reward across trials (Oneshot): {np.mean(trial_oneshot_reward):.3f} ± {np.std(trial_oneshot_reward):.3f}, Average cost across trials (Oneshot): {np.mean(trial_oneshot_cost):.2f} ± {np.std(trial_oneshot_cost):.2f}")
print(f"Average reward across trials (Sequential): {np.mean(trial_seq_reward):.3f} ± {np.std(trial_seq_reward):.3f}, Average cost across trials (Sequential): {np.mean(trial_seq_cost):.2f} ± {np.std(trial_seq_cost):.2f}")

end = time.time()
print("Execution time:", end - start, "seconds")