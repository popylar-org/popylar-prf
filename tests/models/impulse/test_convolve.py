"""Tests for convolution functions."""

import numpy as np
from prfmodel.models.impulse import convolve_prf_impulse_response


def test_convolve_prf_impulse_response():
    """Test that convolve_prf_impulse_response returns response with correct shape."""
    num_batches = 3
    num_prf_frames = 10
    num_irf_frames = 3

    prf_response = np.ones((num_batches, num_prf_frames))
    irf_response = np.ones((num_batches, num_irf_frames))

    resp_conv = convolve_prf_impulse_response(prf_response, irf_response)

    assert resp_conv.shape == (num_batches, num_prf_frames)
