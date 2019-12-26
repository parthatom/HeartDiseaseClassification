import torch
import numpy as np
print("aa")

def gmm_covar(X, means, resps, n_clusters = 2):
    """
    
    Parameters
    ----------
    X shape (n_samples, n_features).

    resp shape (n_samples, n_clusters).

    means shape (n_classes, n_features).

    n_clusters (default 2).
    
    
    Returns
    -------
    Covariance matrix shape (n_features, n_features);
    """
    n_samples = len(X)
    covar_list = []
    for i in range(n_clusters):
        resp = resps[:,i]#.reshape(1, n_samples)
        mean = means[i]
        m = np.sum(resp)
        mean_calculated =  (1/m)*((resp.reshape(n_samples, 1)*X).sum(axis = 0))
        ratio_means = mean/mean_calculated
        print(ratio_means)
        print(ratio_means/X.std(axis = 0))
        print(ratio_means/X.mean(axis = 0))
        print(ratio_means.mean(), ratio_means.std())
        # print(resp.shape, m)
        print(np.exp(mean), mean_calculated)
        a = resp * ((X - mean_calculated).T) # Element wise multiplication  # Shape (n_samples, n_features)
        covar = np.dot(a, (X - mean_calculated))                       # Shape (n_features, n_features)
        covar_list.append(covar)
        covar /= m
    return np.asarray(covar_list)
