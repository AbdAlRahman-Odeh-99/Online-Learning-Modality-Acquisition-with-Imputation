def update_centers(x, cluster, centers, counts, observed_mask):
    """
    Update centers using only observed modalities.

    Args:
        x: sample (M,)
        cluster: cluster label for this sample
        centers: (K,M)
        counts: (K,M)
        observed_mask: (M,)
    """
    for m in range(len(x)):
        if observed_mask[m]:
            counts[cluster, m] += 1
            eta = 1.0 / counts[cluster, m]
            centers[cluster, m] += eta * (x[m] - centers[cluster, m])
    return centers, counts