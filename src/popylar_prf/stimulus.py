"""Containers for stimuli and stimulus grids."""

from collections.abc import Sequence
import numpy as np


class GridDesignShapeError(Exception):
    """
    Exception raised when the shapes of the stimulus design and grid do not match.

    Parameters
    ----------
    design_shape : tuple of int
        Shape of the design array.
    grid_shape : tuple of int
        Shape of the grid array.
    """

    def __init__(self, design_shape: tuple[int], grid_shape: tuple[int]):
        super().__init__(f"Shapes of 'design' {design_shape} and 'grid' {grid_shape} do not match")


class GridDimensionsError(Exception):
    """
    Exception raised when number of grid dimensions except for the last does not match last grid dimension size.

    Parameters
    ----------
    grid_shape: tuple of int
        Shape of the grid array.

    """

    def __init__(self, grid_shape: tuple[int]) -> None:
        num_grid_axes = len(grid_shape[:-1])
        super().__init__(
            f"The number of dimensions in 'grid' {num_grid_axes} does not match its last dimension {grid_shape[-1]}",
        )


class DimensionLabelsError(Exception):
    """
    Exception raised when the number of dimensions does not match the grid's last dimension.

    Parameters
    ----------
    dimensions_len : int
        Length of the dimensions sequence.
    grid_dim : int
        Size of the last dimension of the grid.
    """

    def __init__(self, dimensions_len: int, grid_dim: int):
        super().__init__(f"Length of 'dimensions' {dimensions_len} does not match last dimension of 'grid' {grid_dim}")


class Stimulus:
    """
    Container for a stimulus design and its associated grid.

    Parameters
    ----------
    design : numpy.ndarray
        The stimulus design array containing the stimulus value in one or more dimensions over different time frames.
        The first axis is assumed to be time frames. Additional axes represent design dimensions.
    grid : numpy.ndarray
        The coordinate system of the stimulus array. The last axis is the number of design dimensions
        excluding the time frame dimension. The shape excluding the last axis must match the shape
        of the design excluding the first axis.
    dimension_labels : Sequence[str] or None, optional
        Names of the grid dimensions (e.g., `["y", "x"]`). If given, the number of labels must match the last grid axis.

    Raises
    ------
    GridDesignShapeError
        If the design and grid dimensions do not match.
    GridDimensionsError
        If the number of dimensions of the grid except the last does not match the size of the last grid dimension.
    DimensionLabelsError
        If the number of dimensions does not match the grid's last dimension.

    Notes
    -----
    The shapes of the design and grid must match according to `design.shape[1:] == grid.shape[:-1]`.
    That is, all design dimensions but the first must have the same size as the grid
    dimensions excluding the last grid dimension.

    Examples
    --------
    Create a stimulus on a 2D grid.
    >>> import numpy as np
    >>> from popylar_prf import Stimulus
    >>> num_frames, width, height = 10, 16, 16
    >>> design = np.ones((num_frames, width, height))
    >>> pixel_size = 0.05
    >>> x = (np.arange(width) - (width - 1) / 2) * pixel_size
    >>> y = (np.arange(height) - (height - 1) / 2) * pixel_size
    >>> xv, yv = np.meshgrid(x, y)
    >>> grid = np.stack((xv, yv), axis=-1)  # shape (height, width, 2)
    >>> grid = np.stack((xv, yv), axis=-1)  # shape (height, width, 2)
    >>> # The coordinates of the bottom-left corner:
    >>> grid[0, 0, :]
    array([-0.375, -0.375])
    >>> # The coordinates of the top-right corner:
    >>> grid[15, 15, :]
    array([0.375, 0.375])
    >>> Stimulus(design=design, grid=grid, dimension_labels=["y", "x"])
    Stimulus(design=array[10, 16, 16], grid=array[16, 16, 2], dimension_labels=['y', 'x'])

    """

    __hash__ = None  # We don't want the object to be hashable because it's mutable

    def __init__(self, design: np.ndarray, grid: np.ndarray, dimension_labels: Sequence[str] | None = None):
        self.design = design
        self.grid = grid
        self.dimension_labels = dimension_labels

        self._check_grid_design_shape()
        self._check_grid_dimensions()
        self._check_dimension_labels()

    def _check_grid_design_shape(self) -> None:
        if not self.design.shape[1:] == self.grid.shape[:-1]:
            raise GridDesignShapeError(self.design.shape, self.grid.shape)

    def _check_grid_dimensions(self) -> None:
        if not len(self.grid.shape[:-1]) == self.grid.shape[-1]:
            raise GridDimensionsError(self.grid.shape)

    def _check_dimension_labels(self) -> None:
        if self.dimension_labels is not None and not self.grid.shape[-1] == len(self.dimension_labels):
            raise DimensionLabelsError(len(self.dimension_labels), self.grid.shape[-1])

    def __repr__(self) -> str:
        return f"""{self.__class__.__name__}(
            design={np.array_repr(self.design)},
            grid={np.array_repr(self.grid)},
            dimension_labels={self.dimension_labels}
        )"""

    def __str__(self) -> str:
        design_shape_str = ", ".join([str(s) for s in self.design.shape])
        grid_shape_str = ", ".join([str(s) for s in self.grid.shape])

        return f"{self.__class__.__name__}(design=array[{design_shape_str}], grid=array[{grid_shape_str}], dimension_labels={self.dimension_labels})"  # noqa: E501

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Stimulus):
            msg = "Stimulus objects can only be compared against other Stimulus objects"
            raise TypeError(msg)

        if self.design.shape != other.design.shape or self.grid.shape != other.grid.shape:
            return False

        design_equal = np.all(self.design == other.design)
        grid_equal = np.all(self.grid == other.grid)
        dimensions_labels_equal = self.dimension_labels == other.dimension_labels

        return bool(design_equal and grid_equal and dimensions_labels_equal)
