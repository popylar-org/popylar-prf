"""Containers for stimuli and stimulus grids."""

from collections.abc import Sequence
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


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

    def __init__(self, design_shape: tuple[int], grid_shape: tuple[int, ...]):
        super().__init__(f"Shapes of 'design' {design_shape} and 'grid' {grid_shape} do not match")


class GridDimensionsError(Exception):
    """
    Exception raised when number of grid dimensions except for the last does not match last grid dimension size.

    Parameters
    ----------
    grid_shape: tuple of int
        Shape of the grid array.

    """

    def __init__(self, grid_shape: tuple[int, ...]) -> None:
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


class StimulusDimensionError(Exception):
    """Exception raised when Stimulus does not have the right number of dimensions.

    The dimension for the frames is ignored.

    Parameters
    ----------
    actual : int
        Number of dimensions in the stimulus grid.
    expected : int
        Number of expected dimensions in the stimulus grid.
    """

    def __init__(self, actual: int, expected: int):
        super().__init__(f"Stimulus frames have {actual} dimensions, but expected {expected}.")


class Stimulus:
    """
    Container for a stimulus design and its associated grid.

    Parameters
    ----------
    design : numpy.ndarray
        The stimulus design array containing the stimulus value in one or more dimensions over different time frames.
        The first axis is assumed to be time frames. Additional axes represent design dimensions.
    grid : numpy.ndarray
        The coordinate system of the design. The last axis is the number of design dimensions
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
    >>> stimulus = Stimulus(design=design, grid=grid, dimension_labels=["y", "x"])
    >>> print(stimulus)
    Stimulus(design=array[10, 16, 16], grid=array[16, 16, 2], dimension_labels=['y', 'x'])

    """

    # We don't want the object to be hashable because it's mutable
    __hash__ = None  # type: ignore[assignment]

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

    @classmethod
    def create_2d_bar_stimulus(  # noqa: PLR0913 (too many arguments)
        cls,
        num_frames: int = 100,
        width: int = 128,
        height: int = 128,
        bar_width: int = 20,
        direction: str = "horizontal",
        pixel_size: float = 0.05,
    ) -> "Stimulus":
        """
        Create a bar stimulus that moves across a 2D screen.

        The stimulus starts and ends moving just outside the screen.

        Parameters
        ----------
        num_frames : int, optional
            Number of time frames in the stimulus.
        width : int, optional
            Width of the stimulus grid (in pixels).
        height : int, optional
            Height of the stimulus grid (in pixels).
        bar_width : int, optional
            Width of the moving bar (in pixels).
        direction : {"horizontal", "vertical"}, optional
            Direction in which the bar moves.
        pixel_size : float, optional
            Size of a pixel in spatial units.

        Returns
        -------
        Stimulus
            A Stimulus instance with the generated design and grid.

        Raises
        ------
        ValueError
            If `direction` is not "horizontal" or "vertical".

        Examples
        --------
        >>> stimulus = Stimulus.create_2d_bar_stimulus(num_frames=200)
        >>> print(stimulus)
        Stimulus(design=array[200, 128, 128], grid=array[128, 128, 2], dimension_labels=['y', 'x'])

        """
        # Create a centered grid of x and y coordinates
        x = (np.arange(width) - (width - 1) / 2) * pixel_size
        y = (np.arange(height) - (height - 1) / 2) * pixel_size
        xv, yv = np.meshgrid(x, y)
        grid = np.stack((xv, yv), axis=-1)  # shape (height, width, 2)

        # Create the design array
        design = np.zeros((num_frames, height, width), dtype=np.float32)

        for frame in range(num_frames):
            if direction == "horizontal":
                # Bar moves left to right, starting and ending just outside the screen
                bar_start = int(np.round(-bar_width + frame * (width + bar_width) / (num_frames - 1)))
                bar_end = bar_start + bar_width
                # Only draw within screen bounds
                screen_start = max(bar_start, 0)
                screen_end = min(bar_end, width)

                if screen_start < screen_end:
                    design[frame, :, screen_start:screen_end] = 1.0
            elif direction == "vertical":
                # Bar moves top to bottom, starting and ending just outside the screen
                bar_start = int(np.round(-bar_width + frame * (height + bar_width) / (num_frames - 1)))
                bar_end = bar_start + bar_width
                screen_start = max(bar_start, 0)
                screen_end = min(bar_end, height)

                if screen_start < screen_end:
                    design[frame, screen_start:screen_end, :] = 1.0
            else:
                msg = "Direction must be 'horizontal' or 'vertical'"
                raise ValueError(msg)

        # Dimension y comes first because numpy uses row-major order (i.e., the first axis represents rows or height)
        dimension_labels = ["y", "x"]

        return cls(
            design=design,
            grid=grid,
            dimension_labels=dimension_labels,
        )


def animate_2d_stimulus(
    stimulus: Stimulus,
    title: str | None = None,
    interval: int = 50,
    blit: bool = True,
    repeat_delay: int = 1000,
    **kwargs: Any,  # noqa: ANN401
) -> animation.ArtistAnimation:
    """Animate a 2d stimulus.

    Parameters
    ----------
    stimulus: Stimulus
        The stimulus to visualize.
    title : str or None, Optional.
        Title for the video animation.
    interval : int
        `interval` argument passed to :class:`matplotlib.animation.ArtistAnimation`.
    blit : bool
        `blit` argument passed to :class:`matplotlib.animation.ArtistAnimation`.
    repeat_delay: int
        `repeat_delay` passed to :class:`matplotlib.animation.ArtistAnimation`.
    kwargs : Any
        Additional keyword arguments passed to :class:`matplotlib.animation.ArtistAnimation`.

    Returns
    -------
    A :class:`matplotlib.animation.ArtistAnimation` that can be rendered as video.

    Raises
    ------
    A StimulusDimensionError when `stimulus` is not 2-dimensional.

    Examples
    --------
    >>> from IPython.display import HTML
    >>> from prfmodel.stimulus import Stimulus, make_video
    >>> bar_stimulus = Stimulus.create_2d_bar_stimulus(num_frames=100, width=128, height=64)
    >>> ani = make_2d_animation(bar_stimulus)
    >>> video = ani.to_html5_video()
    >>> HTML(video)
    """
    _verify_dimensions(stimulus, 2)

    font_sizes = {
        "title": 20,
        "labels": 16,
    }

    fig, ax = plt.subplots()
    n_frames = stimulus.design.shape[0]
    grid_extent = _get_grid_extent(stimulus.grid)
    ims = []
    for i in range(n_frames):
        im = ax.imshow(stimulus.design[i, :, :], animated=True, extent=grid_extent, origin="lower")
        ims.append([im])

    if stimulus.dimension_labels:
        ax.set_ylabel(stimulus.dimension_labels[0], fontsize=font_sizes["labels"])
        ax.set_xlabel(stimulus.dimension_labels[1], fontsize=font_sizes["labels"])

    if title:
        ax.set_title(title, fontsize=font_sizes["title"])

    kwargs = kwargs | {"interval": interval, "blit": blit, "repeat_delay": repeat_delay}
    ani = animation.ArtistAnimation(fig, ims, **kwargs)
    plt.close(fig)
    return ani


def _verify_dimensions(stimulus: Stimulus, expected: int) -> None:
    """Verify that stimulus has the right dimensions.

    This checks for the number of dimensions excluding the first (frame) dimension.
    """
    actual = len(stimulus.design.shape)
    if actual != expected + 1:
        raise StimulusDimensionError(actual, expected)


def _get_grid_extent(grid: np.ndarray) -> tuple[float, float, float, float]:
    """From a 2D coordinate grid, return its coordinate limits.

    Output can be passed to :class:`matlplotlib.axes.Axes.imshow`
    """
    left = grid[0, 0, 0]
    bottom = grid[0, 0, -1]

    right = grid[0, -1, 0]
    top = grid[-1, -1, -1]
    return (left, right, bottom, top)
