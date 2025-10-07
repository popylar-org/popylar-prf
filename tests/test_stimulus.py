"""Test stimulus classes."""

import numpy as np
import pytest

# Needs to be imported to recreate stimulus from repr
from numpy import array  # noqa: F401
from prfmodel.stimulus import DimensionLabelsError
from prfmodel.stimulus import GridDesignShapeError
from prfmodel.stimulus import GridDimensionsError
from prfmodel.stimulus import Stimulus


def test_grid_design_shape_error():
    """Check that shape mismatches are detected correctly by Stimulus."""
    with pytest.raises(GridDesignShapeError):
        _ = Stimulus(
            design=np.zeros((1, 2)),
            grid=np.zeros((1, 1)),
        )


def test_grid_dimension_error():
    """Check that shape mismatches are detected correctly by Stimulus."""
    with pytest.raises(GridDimensionsError):
        _ = Stimulus(
            design=np.zeros((1, 1, 2)),
            grid=np.zeros((1, 2, 1)),
        )


def test_dimension_labels_error():
    """Check that dimension mismatches are detected correctly by Stimulus."""
    with pytest.raises(DimensionLabelsError):
        _ = Stimulus(
            design=np.zeros((1, 2)),
            grid=np.zeros((2, 1)),
            dimension_labels=["x", "y"],
        )


@pytest.fixture
def stimulus():
    """Stimulus object."""
    return Stimulus(
        design=np.zeros((1, 2, 1)),
        grid=np.zeros((2, 1, 2)),
        dimension_labels=["x", "y"],
    )


def test_repr(stimulus: Stimulus):
    """Test machine-readable string representation of Stimulus."""
    stimulus_2 = eval(repr(stimulus))  # noqa: S307

    assert stimulus == stimulus_2


def test_str(stimulus: Stimulus):
    """Test human-readable string representation of Stimulus."""
    assert str(stimulus) == "Stimulus(design=array[1, 2, 1], grid=array[2, 1, 2], dimension_labels=['x', 'y'])"


def test_eq(stimulus: Stimulus):
    """Test equality of two Stimulus objects."""
    stimulus_2 = Stimulus(
        design=np.zeros((1, 2, 1)),
        grid=np.zeros((2, 1, 2)),
        dimension_labels=["x", "y"],
    )

    assert stimulus == stimulus_2

    with pytest.raises(TypeError):
        _ = stimulus == np.zeros((0, 0, 0))


def test_ne(stimulus: Stimulus):
    """Test inequality of two Stimulus objects."""
    stimulus_3 = Stimulus(
        design=np.zeros((1, 2, 2)),
        grid=np.zeros((2, 2, 2)),
        dimension_labels=["x", "y"],
    )

    assert stimulus != stimulus_3


def test_hash(stimulus: Stimulus):
    """Test hash of two Stimulus objects."""
    with pytest.raises(TypeError):
        _ = hash(stimulus)


@pytest.mark.parametrize(
    ("dimensions", "axis"),
    [("1D", "width"), ("2D", "width"), ("2D", "height"), ("3D", "depth")],
)
def test_rectangular_grid(dimensions: str, axis: str):
    """Check that rectangular grids can be created."""
    width = 10
    height = 20
    depth = 10

    scale = 2

    match axis:
        case "width":
            width *= scale
        case "height":
            height *= scale
        case "depth":
            depth *= scale

    num_frames = 10

    # 1D shape
    design_shape = (num_frames, width)
    grid_shape = (width, 1)

    match dimensions:
        case "2D":
            design_shape = (num_frames, width, height)
            grid_shape = (width, height, 2)
        case "3D":
            design_shape = (num_frames, width, height, depth)
            grid_shape = (width, height, depth, 3)

    design = np.zeros(design_shape)
    grid = np.zeros(grid_shape)

    stimulus = Stimulus(
        design=design,
        grid=grid,
    )

    assert isinstance(stimulus, Stimulus)


@pytest.mark.parametrize("direction", ["horizontal", "vertical"])
def test_create_2d_bar_stimulus(direction: str):
    """Check that a valid 2D bar stimulus can be created."""
    stimulus = Stimulus.create_2d_bar_stimulus(
        num_frames=10,
        width=5,
        height=4,
        direction=direction,
    )

    assert isinstance(stimulus, Stimulus)
    # First and last frame should be empty
    assert np.all(stimulus.design[0] == 0.0)
    assert np.all(stimulus.design[-1] == 0.0)

    # Design should have range 0-1
    assert np.min(stimulus.design) == 0.0
    assert np.max(stimulus.design) == 1.0

    match direction:
        case "horizontal":
            # For horizontal bars, each row (height axis) should have the same value within a frame
            # Each row in the frame should be constant
            rows_equal = np.all(stimulus.design == stimulus.design[:, [0], :])  # (10,5,4) == (10,1,4)
            assert np.all(rows_equal)
        case "vertical":
            # For vertical bars, each column (width axis) should have the same value within a frame
            # All values in each column should be equal for all frames
            cols_equal = np.all(stimulus.design == stimulus.design[:, :, [0]])
            assert np.all(cols_equal)
