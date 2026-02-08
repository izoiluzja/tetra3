"""Test suite for utility functions in tetra3."""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tetra3.tetra3 as tetra3_module

_insert_at_index = tetra3_module._insert_at_index
_get_table_index_from_hash = tetra3_module._get_table_index_from_hash
_key_to_index = tetra3_module._key_to_index
_compute_vectors = tetra3_module._compute_vectors
_compute_centroids = tetra3_module._compute_centroids
_undistort_centroids = tetra3_module._undistort_centroids
_distort_centroids = tetra3_module._distort_centroids
_find_rotation_matrix = tetra3_module._find_rotation_matrix
_find_centroid_matches = tetra3_module._find_centroid_matches


def test_insert_at_index():
    """Test inserting a pattern into a table."""
    table = np.zeros((10, 5), dtype=np.uint64)
    pattern = np.array([1, 2, 3, 4, 5], dtype=np.uint64)
    hash_index = 2
    index = _insert_at_index(pattern, hash_index, table)
    assert np.array_equal(table[index], pattern)


def test_get_table_index_from_hash():
    """Test retrieving indices from a hash."""
    table = np.zeros((10, 5), dtype=np.uint64)
    pattern = np.array([1, 2, 3, 4, 5], dtype=np.uint64)
    hash_index = 2
    _insert_at_index(pattern, hash_index, table)
    indices = _get_table_index_from_hash(hash_index, table)
    assert len(indices) == 1
    assert indices[0] == hash_index


def test_key_to_index():
    """Test converting a key to a hash index."""
    key = np.array([1, 2, 3, 4, 5], dtype=np.uint64)
    bin_factor = 50
    max_index = 1000
    hash_index = _key_to_index(key, bin_factor, max_index)
    assert isinstance(hash_index, np.uint64)
    assert hash_index < max_index


def test_compute_vectors():
    """Test computing vectors from centroids."""
    centroids = np.array([[100, 100], [200, 200], [300, 300]], dtype=np.float32)
    size = (400, 400)
    fov = np.deg2rad(10)
    vectors = _compute_vectors(centroids, size, fov)
    assert vectors.shape == (3, 3)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_compute_centroids():
    """Test computing centroids from vectors."""
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    size = (400, 400)
    fov = np.deg2rad(10)
    centroids, indices = _compute_centroids(vectors, size, fov, trim=True)
    assert centroids.shape[0] <= 3
    assert len(indices) == centroids.shape[0]


def test_undistort_centroids():
    """Test undistorting centroids."""
    centroids = np.array([[100, 100], [200, 200], [300, 300]], dtype=np.float32)
    size = (400, 400)
    k = -0.1
    undistorted = _undistort_centroids(centroids, size, k)
    assert undistorted.shape == (3, 2)


def test_distort_centroids():
    """Test distorting centroids."""
    centroids = np.array([[100, 100], [200, 200], [300, 300]], dtype=np.float32)
    size = (400, 400)
    k = 0.1
    distorted = _distort_centroids(centroids, size, k)
    assert distorted.shape == (3, 2)


def test_find_rotation_matrix():
    """Test finding the rotation matrix between two sets of vectors."""
    image_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    catalog_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    rotation_matrix = _find_rotation_matrix(image_vectors, catalog_vectors)
    assert rotation_matrix.shape == (3, 3)
    assert np.allclose(rotation_matrix, np.eye(3))


def test_find_centroid_matches():
    """Test finding matches between image and catalog centroids."""
    image_centroids = np.array([[100, 100], [200, 200], [300, 300]], dtype=np.float32)
    catalog_centroids = np.array([[100, 100], [200, 200], [300, 300]], dtype=np.float32)
    r = 10
    matches = _find_centroid_matches(image_centroids, catalog_centroids, r)
    assert matches.shape[0] == 3
    assert np.array_equal(matches[:, 0], matches[:, 1])
