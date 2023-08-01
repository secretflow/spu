import numpy as np
def _check_sample_weight(X, sample_weight):
    """Set sample_weight if None, and check for correct dtype"""
    n_samples = X.shape[0]
    if sample_weight is None:
        return np.ones(n_samples, dtype=X.dtype)
    else:
        sample_weight = np.asarray(sample_weight)
        if n_samples != len(sample_weight):
            raise ValueError("n_samples=%d should be == len(sample_weight)=%d"
                             % (n_samples, len(sample_weight)))
        # normalize the weights to sum up to n_samples
        scale = n_samples / sample_weight.sum()
        return (sample_weight * scale).astype(X.dtype)