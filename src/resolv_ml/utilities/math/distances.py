# TODO - DOC

import keras.ops as k_ops


def compute_pairwise_distance_matrix(x):
    x_expanded = k_ops.expand_dims(x, 1)  # Expand dimensions to allow broadcasting
    x_transposed = k_ops.transpose(x_expanded, axis=(0, 2, 1))
    distance_matrix = x_expanded - x_transposed  # Compute pairwise differences
    return distance_matrix


def minkowski_distance(x1, x2, p):
    """
    Computes the Minkowski distance between two tensors of the same shape.

    Arguments:
    x1 -- First tensor, shape (batch_size, n, d)
    x2 -- Second tensor, shape (batch_size, m, d)
    p -- Parameter for the Minkowski distance.

    Returns:
    distances -- Pairwise distances, shape (batch_size, n, m)
    """
    # Expand dimensions to enable broadcasting
    x1_expand = k_ops.expand_dims(x1, axis=2)
    x2_expand = k_ops.expand_dims(x2, axis=1)
    diff = k_ops.abs(k_ops.subtract(x1_expand, x2_expand))
    distances = k_ops.power(k_ops.sum(k_ops.power(diff, p), axis=-1), k_ops.divide(1.0, p))
    return distances


def dot_product_distance(x1, x2, normalize: bool = False):
    """
    Computes the dot product distance between two tensors of the same shape.

    Arguments:
    x1 -- First tensor, shape (batch_size, n, d)
    x2 -- Second tensor, shape (batch_size, m, d)
    normalize -- If true compute the cosine distance

    Returns:
    distances -- Pairwise distances, shape (batch_size, n, m)
    """
    x2_transposed = k_ops.transpose(x2, axes=(0, 2, 1))
    dot_product = k_ops.dot(x1, x2_transposed, axes=(2, 1), normalize=normalize)
    distances = k_ops.subtract(1.0, dot_product)
    return distances


def cosine_distance(x1, x2):
    """
    Computes the cosine distance between two tensors of the same shape.

    Arguments:
    x1 -- First tensor, shape (batch_size, n, d)
    x2 -- Second tensor, shape (batch_size, m, d)

    Returns:
    distances -- Pairwise distances, shape (batch_size, n, m)
    """
    distances = dot_product_distance(x1, x2, normalize=True)
    return distances


def sqrt_euclidean_distance(x1, x2):
    """
    Computes the square root Euclidean distance between two tensors of the same shape.

    Arguments:
    x1 -- First tensor, shape (batch_size, n, d)
    x2 -- Second tensor, shape (batch_size, m, d)

    Returns:
    distances -- Pairwise distances, shape (batch_size, n, m)
    """
    # Expand dimensions to enable broadcasting
    x1_expand = k_ops.expand_dims(x1, axis=2)
    x2_expand = k_ops.expand_dims(x2, axis=1)
    squared_diff = k_ops.square(k_ops.subtract(x1_expand, x2_expand))
    sum_squared_diff = k_ops.sum(squared_diff, axis=-1)
    distances = k_ops.sqrt(sum_squared_diff)
    return distances


def pairwise_distance(x1, x2, metric='euclidean'):
    """
    Computes pairwise distances between two tensors of the same shape.

    Arguments:
    x1 -- First tensor, shape (batch_size, n, d)
    x2 -- Second tensor, shape (batch_size, m, d)
    metric -- Distance metric to use (default: 'euclidean')

    Returns:
    distances -- Pairwise distances, shape (batch_size, n, m)
    """
    if metric == 'euclidean':
        distances = minkowski_distance(x1, x2, p=2)
    elif metric == 'manhattan':
        distances = minkowski_distance(x1, x2, p=1)
    elif metric == 'cosine':
        distances = cosine_distance(x1, x2)
    elif metric == 'sqrt_euclidean':
        distances = sqrt_euclidean_distance(x1, x2)
    elif metric == 'dot_product':
        distances = dot_product_distance(x1, x2)
    else:
        raise ValueError('Metric {} not supported. Available options: euclidean, cosine, manhattan, '
                         'sqrt_euclidean, dot_product'.format(metric))
    return distances
