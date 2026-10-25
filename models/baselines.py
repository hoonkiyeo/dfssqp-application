"""
Existing fixed comparisons; all directions/coefficients use training data only.
Mean/PCA are industry-summary controls (application background: He et al., 2023,
doi:10.1016/j.ribaf.2023.101983). Our neural link/training are application choices.
Pass either fixed direction to models.objective.evaluate for the unchanged fit.
"""

import numpy as np

def fixed_directions(training):
    """
    Equal weights and the first principal direction, both with unit norm.
    """    
    z = np.asarray(training['z'], dtype=np.float64)
    if z.ndim != 2 or min(z.shape) < 1 or len(z) < 2 or not np.isfinite(z).all():
        raise ValueError('PCA needs a finite training matrix with at least two rows.')

    # Mean: equal weights. The shared evaluator standardizes the resulting index.
    mean = np.ones(z.shape[1]) / np.sqrt(z.shape[1])

    # PCA: center each industry over training months; do not rescale its variance.
    _, _, right = np.linalg.svd(z - z.mean(axis=0), full_matrices=False)
    pca = right[0].copy()
    if pca[np.argmax(np.abs(pca))] < 0:
        pca *= -1  # Fix the arbitrary SVD sign, exactly as in the original code.
    return {'mean': mean, 'pca': pca}


def aggregate_only(training, validation):
    """
    Fit y = a + beta*u by OLS. Return predictions/MSE in original units.
    """
    tu, ty = (np.asarray(training[key], dtype=np.float64) for key in ('u', 'y'))
    vu, vy = (np.asarray(validation[key], dtype=np.float64) for key in ('u', 'y'))
    for u, y in ((tu, ty), (vu, vy)):
        if u.ndim != 1 or u.shape != y.shape or len(u) == 0 or not np.isfinite([u, y]).all():
            raise ValueError('Aggregate-only needs matching, nonempty, finite u/y vectors.')

    x = np.column_stack([np.ones(len(ty)), tu])
    coefficients, _, rank, _ = np.linalg.lstsq(x, ty, rcond=None)
    training_predictions = x @ coefficients
    validation_predictions = coefficients[0] + coefficients[1] * vu
    return {'coefficients': coefficients, 'design_rank': int(rank),
            'training_predictions': training_predictions,
            'validation_predictions': validation_predictions,
            'training_mse': float(np.mean((training_predictions - ty)**2)),
            'mse': float(np.mean((validation_predictions - vy)**2))}
