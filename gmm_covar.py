import torch
import numpy as np

print('hahah')

print('hello from anyfile notepad.')
print(' this looks cool, although it is pretty useless')

print(' I also need to figure out if I can directly save both in drive and github ')

print(' Okay lets start ')

print(' firstly i need func that can take X, prob, mean and calculate the covar ')
print('edited T')

def gmm_covar(X, means, resps, n_clusters = 2):
    """
    X shape (n_samples, n_features)
    resp shape (n_samples, n_clusters)
    means shape (n_classes, n_features)
    n_clusters default 2
    returns Covariance matrix shape (n_features, n_features)
    """
    n_samples = len(X)
    covar_list = []
    for i in range(n_clusters):
        resp = resps[:,i].reshape(1, n_samples)
        mean = means[i]
        m = np.sum(resp)
        print(len(resp), m)
        a = resp * ((X - mean).T) # Element wise multiplication  # Shape (n_samples, n_features)
        covar = np.dot(a, (X - mean))                       # Shape (n_features, n_features)
        covar_list.append(covar)
        covar /= m
    return np.asarray(covar_list)
