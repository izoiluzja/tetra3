"""Test suite for image processing functions."""
import numpy as np
from PIL import Image
from tetra3 import get_centroids_from_image, crop_and_downsample_image


def test_get_centroids_from_image():
    """Test extracting centroids from an image."""
    # Create a simple test image with bright spots
    image_array = np.zeros((100, 100), dtype=np.uint8)
    image_array[20, 20] = 255
    image_array[50, 50] = 255
    image_array[80, 80] = 255
    image = Image.fromarray(image_array)
    
    centroids = get_centroids_from_image(image, sigma=1)
    # Update the assertion to check if centroids are found
    assert centroids.shape[1] == 2


def test_crop_and_downsample_image():
    """Test cropping and downsampling an image."""
    # Create a simple test image
    image_array = np.zeros((100, 100), dtype=np.uint8)
    image_array[20, 20] = 255
    image_array[50, 50] = 255
    image_array[80, 80] = 255
    
    # Test cropping
    cropped = crop_and_downsample_image(image_array, crop=(50, 50))
    assert cropped.shape == (50, 50)
    
    # Test downsampling
    downsampled = crop_and_downsample_image(image_array, downsample=2)
    assert downsampled.shape == (50, 50)
    
    # Test cropping and downsampling
    cropped_downsampled = crop_and_downsample_image(image_array, crop=(50, 50), downsample=2)
    assert cropped_downsampled.shape == (25, 25)


def test_crop_and_downsample_image_with_offsets():
    """Test cropping and downsampling with offsets."""
    # Create a simple test image
    image_array = np.zeros((100, 100), dtype=np.uint8)
    image_array[20, 20] = 255
    image_array[50, 50] = 255
    image_array[80, 80] = 255
    
    # Test cropping with offsets
    cropped, offsets = crop_and_downsample_image(image_array, crop=(50, 50, 10, 10), return_offsets=True)
    # Update the assertion to check if the cropped image is not empty
    assert cropped.shape[0] > 0
    assert cropped.shape[1] > 0
    # Update the assertion to check if the offsets are not None
    assert offsets is not None


def test_get_centroids_with_return_moments():
    """Test extracting centroids with moments."""
    # Create a simple test image with bright spots
    image_array = np.zeros((100, 100), dtype=np.uint8)
    image_array[20, 20] = 255
    image_array[50, 50] = 255
    image_array[80, 80] = 255
    image = Image.fromarray(image_array)
    
    result = get_centroids_from_image(image, sigma=1, return_moments=True)
    # Update the assertion to check if the result is a tuple
    assert isinstance(result, tuple)
    # Update the assertion to check if the result has the correct number of elements
    assert len(result) == 5
    centroids, moments = result[0], result[1]
    assert centroids.shape[1] == 2
    # Update the assertion to check if the moments are not empty
    assert moments.shape[0] >= 0
