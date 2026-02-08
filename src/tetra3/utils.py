# This file contains utility functions that are used in the solver.py file.

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cdist
import itertools

_MAGIC_RAND = np.uint64(2654435761)

def _insert_at_index(pattern, hash_index, table):
    """Inserts to table with quadratic probing. Returns table index where pattern was inserted."""
    max_ind = np.uint64(table.shape[0])
    hash_index = np.uint64(hash_index)
    for c in itertools.count():
        c = np.uint64(c)
        i = (hash_index + c*c) % max_ind
        if all(table[i, :] == 0):
            table[i, :] = pattern
            return i

def _get_table_index_from_hash(hash_index, table):
    """Gets from table with quadratic probing, returns list of all possibly matching indices."""
    max_ind = np.uint64(table.shape[0])
    hash_index = np.uint64(hash_index)
    found = []
    for c in itertools.count():
        c = np.uint64(c)
        i = (hash_index + c*c) % max_ind
        if all(table[i, :] == 0):
            return np.array(found)
        else:
            found.append(i)


def _key_to_index(key, bin_factor, max_index):
    """Get hash index for a given key. Can be length p list or n by p array."""
    key = np.uint64(key)
    bin_factor = np.uint64(bin_factor)
    max_index = np.uint64(max_index)
    # If p is the length of the key (default 5) and B is the number of bins (default 50,
    # calculated from max error), this will first give each key a unique index from
    # 0 to B^p-1, then multiply by large number and modulo to max index to randomise.
    if key.ndim == 1:
        hash_indices = np.sum(key*bin_factor**np.arange(len(key), dtype=np.uint64),
                              dtype=np.uint64)
    else:
        hash_indices = np.sum(key*bin_factor**np.arange(key.shape[1], dtype=np.uint64)[None, :],
                              axis=1, dtype=np.uint64)
    with np.errstate(over='ignore'):
        hash_indices = (hash_indices*_MAGIC_RAND) % max_index
    return hash_indices


def _compute_vectors(centroids, size, fov):
    """Get unit vectors from star centroids (pinhole camera)."""
    # compute list of (i,j,k) vectors given list of (y,x) star centroids and
    # an estimate of the image's field-of-view in the x dimension
    # by applying the pinhole camera equations
    centroids = np.array(centroids, dtype=np.float32)
    (height, width) = size[:2]
    scale_factor = np.tan(fov/2)/width*2
    star_vectors = np.ones((len(centroids), 3))
    # Pixel centre of image
    img_center = [height/2, width/2]
    # Calculate normal vectors
    star_vectors[:, 2:0:-1] = (img_center - centroids) * scale_factor
    star_vectors = star_vectors / norm(star_vectors, axis=1)[:, None]
    return star_vectors


def _compute_centroids(vectors, size, fov, trim=True):
    """Get (undistorted) centroids from a set of (derotated) unit vectors
    vectors: Nx3 of (i,j,k) where i is boresight, j is x (horizontal)
    size: (height, width) in pixels.
    fov: horizontal field of view in radians.
    trim: only keep ones within the field of view, also returns list of indices kept
    """
    (height, width) = size[:2]
    scale_factor = -width/2/np.tan(fov/2)
    centroids = scale_factor*vectors[:, 2:0:-1]/vectors[:, [0]]
    centroids += [height/2, width/2]
    if not trim:
        return centroids
    else:
        keep = np.flatnonzero(np.logical_and(
            np.all(centroids > [0, 0], axis=1),
            np.all(centroids < [height, width], axis=1)))
        return (centroids[keep, :], keep)


def _distort_centroids(centroids, size, k, tol=1e-6, maxiter=30):
    """Distort centroids corresponding to r_u = r_d(1 - k'*r_d^2)/(1 - k),
    where k'=k*(2/width)^2 i.e. k is the distortion that applies
    width/2 away from the centre.

    Iterates with Newton-Raphson until the step is smaller than tol
    or maxiter iterations have been exhausted.
    """
    centroids = np.array(centroids, dtype=np.float32)
    (height, width) = size[:2]
    # Centre
    centroids -= [height/2, width/2]
    r_undist = norm(centroids, axis=1)/width*2
    # Initial guess, distorted are the same positon
    r_dist = r_undist.copy()
    for i in range(maxiter):
        r_undist_est = r_dist*(1 - k*r_dist**2)/(1 - k)
        dru_drd = (1 - 3*k*r_dist**2)/(1 - k)
        error = r_undist - r_undist_est
        r_dist += error/dru_drd

        if np.all(np.abs(error) < tol):
            break

    centroids *= (r_dist/r_undist)[:, None]
    centroids += [height/2, width/2]
    return centroids

    
def _undistort_centroids(centroids, size, k):
    """Apply r_u = r_d(1 - k'*r_d^2)/(1 - k) undistortion, where k'=k*(2/width)^2,
    i.e. k is the distortion that applies width/2 away from the centre.
    centroids: Nx2 pixel coordinates (y, x), (0.5, 0.5) top left pixel centre.
    size: (height, width) in pixels.
    k: distortion, negative is barrel, positive is pincushion
    """
    centroids = np.array(centroids, dtype=np.float32)
    (height, width) = size[:2]
    # Centre
    centroids -= [height/2, width/2]
    # Scale
    scale = (1 - k*(norm(centroids, axis=1)/width*2)**2)/(1 - k)
    centroids *= scale[:, None]
    # Decentre
    centroids += [height/2, width/2]
    return centroids
    

def _find_rotation_matrix(image_vectors, catalog_vectors):
    """Calculate the least squares best rotation matrix between the two sets of vectors.
    image_vectors and catalog_vectors both Nx3. Must be ordered as matching pairs.
    """
    # find the covariance matrix H between the image and catalog vectors
    H = np.dot(image_vectors.T, catalog_vectors)
    # use singular value decomposition to find the rotation matrix
    (U, S, V) = np.linalg.svd(H)
    return np.dot(U, V)


def _find_centroid_matches(image_centroids, catalog_centroids, r):
    """Find matching pairs, unique and within radius r
    image_centroids: Nx2 (y, x) in pixels
    catalog_centroids: Mx2 (y, x) in pixels
    r: radius in pixels

    returns Kx2 list of matches, first colum is index in image_centroids,
        second column is index in catalog_centroids
    """
    dists = cdist(image_centroids, catalog_centroids)
    matches = np.argwhere(dists < r)
    # Make sure we only have unique 1-1 matches
    matches = matches[np.unique(matches[:, 0], return_index=True)[1], :]
    matches = matches[np.unique(matches[:, 1], return_index=True)[1], :]
    return matches
