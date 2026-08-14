import numpy as np
import pytest

N_LANDMARKS = 478


@pytest.fixture
def synthetic_landmarks() -> np.ndarray:
    """478 face-mesh-like landmark points spread over a 400x400 image."""
    rng = np.random.default_rng(42)
    pts = rng.uniform(50, 350, size=(N_LANDMARKS, 2))
    return pts.astype(np.float64)
