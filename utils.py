import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import linear_sum_assignment


def generate_view_combinations(m_modalities):
    """
    Generate all view combinations with modality 1 always included.

    Returns:
        List of tuples representing modality indices.
    """
    modalities = list(range(2, m_modalities + 1))   # modalities after first
    all_views = []

    for r in range(len(modalities) + 1):
        for combo in combinations(modalities, r):
            view = (1,) + combo
            all_views.append(view)

    return all_views

def generate_combination_costs(view_combinations, cost_per_modality):
    """
    Compute total cost for each modality combination.

    Args:
        view_combinations: list of tuples
            e.g. [(1,), (1,2), (1,3), (1,2,3)]

        cost_per_modality: dict
            e.g. {'1':0, '2':1, '3':1}

    Returns:
        costs: dict
            maps combination -> total cost
    """
    costs = {}

    for combo in view_combinations:
        total_cost = sum(cost_per_modality[str(mod)] for mod in combo)
        costs[combo] = total_cost

    return costs

def print_true_learned_means(m_modalities, true_means, learned_centers):
    print(f"True means: ")
    for m in range(m_modalities):
        print(f"Modality {m+1}: {true_means[m].flatten()}")
    print(f"Learned means: ")
    for m in range(m_modalities):
        print(f"Modality {m+1}: {learned_centers[:, m]}")

def combo_to_mask(combo, m_modalities):
    mask = np.zeros(m_modalities, dtype=bool)
    mask[np.array(combo) - 1] = True
    return mask

def match_cluster_labels(learned_centers, true_centers):
    # Convert dict of arrays (modality -> (K,1)) to (K,M)
    true_mean_array = np.hstack([true_centers[m] for m in sorted(true_centers.keys())])
    K = learned_centers.shape[0]
    
    # Cost matrix: squared Euclidean distances
    cost_matrix = np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            cost_matrix[i,j] = np.sum((learned_centers[i] - true_mean_array[j])**2)
    
    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    return mapping

def posterior_probability(x, centers, observed_mask, sigma=1.0):
    """
    Calculate the posterior probability of each cluster given observed data.
    """
    K, M = centers.shape
    obs_indices = np.where(observed_mask)[0]
    
    # If nothing is observed, return uniform weights
    if len(obs_indices) == 0:
        return np.ones(K) / K

    # Calculate squared Euclidean distance on observed dimensions only
    diff = x[obs_indices] - centers[:, obs_indices]
    sq_dist = np.sum(diff**2, axis=1)
    
    # Softmax with a negative sign (closer = higher probability)
    # Using a small epsilon to avoid division by zero
    log_probs = -0.5 * sq_dist / (sigma**2)
    
    # Numerical stability trick: subtract max before exp
    shifted_log_probs = log_probs - np.max(log_probs)
    probs = np.exp(shifted_log_probs)
    
    return probs / np.sum(probs)

def plot_combo_selection(combo_selection_history, view_combinations, title, save_path=None):
    """
    Cumulative selection count for each combo over time.
    
    combo_selection_history: list of chosen combos at each timestep
    view_combinations:       list of all possible combos
    """
    df = pd.DataFrame(
        {str(combo): [1 if c == combo else 0 for c in combo_selection_history]
         for combo in view_combinations}
    )
    
    df_cumsum = df.cumsum()

    fig, ax = plt.subplots(figsize=(12, 5))
    for combo in view_combinations:
        ax.plot(df_cumsum[str(combo)], label=str(combo))

    ax.set_xlabel("Horizon")
    ax.set_ylabel("Cumulative selection count")
    ax.set_title(title)
    ax.legend(title="Combo", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    #plt.show()