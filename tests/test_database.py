"""Test suite for database operations."""
import numpy as np
import pytest
from pathlib import Path
from tetra3 import Tetra3


def test_generate_database(tmp_path):
    """Test generating a database."""
    t3 = Tetra3(load_database=None)
    save_path = tmp_path / 'test_database'
    
    # Skip the test if the star catalog is not available
    import pytest
    try:
        # Generate a small database for testing
        t3.generate_database(max_fov=10, min_fov=5, save_as=save_path)
        
        assert save_path.with_suffix('.npz').exists()
        
        # Load the generated database
        t3.load_database(save_path)
        assert t3.has_database is True
    except AssertionError as e:
        if "No star catalogue found" in str(e):
            pytest.skip("Star catalog not found, skipping test")
        else:
            raise


def test_load_nonexistent_database():
    """Test loading a non-existent database."""
    t3 = Tetra3(load_database=None)
    with pytest.raises(FileNotFoundError):
        t3.load_database('nonexistent_database')


def test_database_properties():
    """Test accessing database properties."""
    t3 = Tetra3(load_database='default_database')
    props = t3.database_properties
    
    assert 'max_fov' in props
    assert 'min_fov' in props
    assert 'star_catalog' in props
    assert 'pattern_size' in props
    assert 'pattern_bins' in props


def test_save_and_load_database(tmp_path):
    """Test saving and loading a database."""
    t3 = Tetra3(load_database='default_database')
    save_path = tmp_path / 'test_database'
    
    # Save the database
    t3.save_database(save_path)
    assert save_path.with_suffix('.npz').exists()
    
    # Create a new instance and load the saved database
    t3_new = Tetra3(load_database=None)
    t3_new.load_database(save_path)
    
    # Compare properties
    assert t3_new.database_properties['max_fov'] == t3.database_properties['max_fov']
    assert t3_new.database_properties['min_fov'] == t3.database_properties['min_fov']
