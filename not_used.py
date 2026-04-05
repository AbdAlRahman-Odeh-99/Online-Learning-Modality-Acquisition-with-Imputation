# def compute_cluster_posterior(x_sample, centers, observed_mask):
#     """
#     Compute posterior probability over clusters using observed modalities only.

#     Parameters
#     ----------
#     x_sample : array shape (M,)
#         Single sample

#     centers : array shape (K, M)
#         Learned centers

#     observed_mask : bool array shape (M,)
#         True for observed modalities

#     Returns
#     -------
#     posterior : array shape (K,)
#     """

#     observed_x = x_sample[observed_mask]
#     observed_centers = centers[:, observed_mask]

#     # Identity covariance => squared Euclidean distance
#     distances = np.sum((observed_centers - observed_x) ** 2, axis=1)

#     # Gaussian likelihood proportional to exp(-0.5 * distance)
#     likelihoods = np.exp(-0.5 * distances)

#     posterior = likelihoods / np.sum(likelihoods)

#     return posterior

# def impute_missing_views(x_sample, centers, observed_mask):
#     """
#     Impute missing modalities using posterior-weighted learned centers.

#     Parameters
#     ----------
#     x_sample : array shape (M,)
#     centers : array shape (K, M)
#     observed_mask : bool array shape (M,)

#     Returns
#     -------
#     x_completed : array shape (M,)
#     posterior : array shape (K,)
#     """

#     posterior = compute_cluster_posterior(x_sample, centers, observed_mask)

#     x_completed = x_sample.copy()

#     missing_mask = ~observed_mask

#     # posterior-weighted mean for missing modalities
#     x_completed[missing_mask] = np.sum(
#         posterior[:, None] * centers[:, missing_mask],
#         axis=0
#     )

#     return x_completed, posterior