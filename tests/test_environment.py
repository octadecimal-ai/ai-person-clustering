#!/usr/bin/env python3
"""
Environment Compatibility Tests for Apple Silicon M1
AI Person Clustering System

This test verifies that all critical dependencies work correctly
on Apple Silicon M1 with MPS support.
"""

import pytest
import platform
import torch
import numpy as np


def test_system_info():
    """Test system platform and Python version."""
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    
    # Verify we're on Apple Silicon
    if platform.system() == "Darwin":
        assert platform.machine() == "arm64", "Should be running on Apple Silicon (arm64)"


def test_pytorch_mps():
    """Test PyTorch MPS functionality on Apple Silicon."""
    print(f"PyTorch version: {torch.__version__}")
    
    # MPS should be available and built
    assert torch.backends.mps.is_available(), "MPS should be available on Apple Silicon"
    assert torch.backends.mps.is_built(), "MPS should be built in this PyTorch version"
    
    # Test basic MPS operations
    device = torch.device("mps")
    
    # Create tensors on MPS device
    x = torch.tensor([1.0, 2.0, 3.0]).to(device)
    y = torch.tensor([4.0, 5.0, 6.0]).to(device)
    
    # Perform computation on MPS
    result = x + y
    expected = torch.tensor([5.0, 7.0, 9.0])
    
    # Move back to CPU for comparison
    assert torch.allclose(result.cpu(), expected), "MPS computation should work correctly"
    
    print("✅ MPS operations working correctly!")


def test_yolo_import():
    """Test YOLO import and basic functionality."""
    from ultralytics import YOLO
    
    # This should not raise any exceptions
    print("✅ Ultralytics YOLO imported successfully")


def test_face_recognition():
    """Test face recognition library import."""
    import face_recognition
    import dlib
    
    print(f"✅ face_recognition imported (dlib version available)")


def test_mediapipe():
    """Test MediaPipe import and basic functionality."""
    import mediapipe as mp
    
    # Test basic MediaPipe component
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    print("✅ MediaPipe imported with pose solutions")


def test_opencv():
    """Test OpenCV import and basic functionality."""
    import cv2
    
    print(f"✅ OpenCV version: {cv2.__version__}")
    
    # Test basic image operations
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    assert gray.shape == (100, 100), "OpenCV image processing should work"


def test_clustering_libraries():
    """Test clustering and ML libraries."""
    import sklearn
    import faiss
    import umap
    import hdbscan
    import pandas as pd
    import matplotlib.pyplot as plt
    
    print("✅ All clustering libraries imported successfully")
    
    # Test basic clustering functionality
    from sklearn.cluster import DBSCAN
    from sklearn.datasets import make_blobs
    
    # Generate sample data
    X, _ = make_blobs(n_samples=20, centers=3, random_state=42)
    
    # Test DBSCAN
    clustering = DBSCAN(eps=1.0, min_samples=2).fit(X)
    
    assert len(clustering.labels_) == 20, "DBSCAN should process all samples"
    print("✅ DBSCAN clustering test passed")


def test_development_tools():
    """Test development and testing tools."""
    import pytest
    import black
    import flake8
    
    print("✅ Development tools imported successfully")


if __name__ == "__main__":
    print("=" * 70)
    print("🧪 AI PERSON CLUSTERING - ENVIRONMENT COMPATIBILITY TESTS")
    print("=" * 70)
    
    # Run all tests
    test_system_info()
    test_pytorch_mps() 
    test_yolo_import()
    test_face_recognition()
    test_mediapipe()
    test_opencv()
    test_clustering_libraries()
    test_development_tools()
    
    print("=" * 70)
    print("🎉 ALL TESTS PASSED! Environment is Apple Silicon M1 compatible!")
    print("=" * 70) 