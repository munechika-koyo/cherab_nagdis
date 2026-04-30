from raysect.core.math import AffineMatrix3D
from raysect.core.scenegraph._nodebase import _NodeBase
from raysect.optical.observer.base import Observer2D, Pipeline2D

class ThinLensCCDArray(Observer2D):
    """An ideal CCD-like imaging sensor that simulates a thin lens.

    The CCD is a regular array of square pixels. Each pixel samples red, green,
    and blue channels (behaving like a Foveon imaging sensor). The sensor width
    is set by the `width` parameter. The sensor height is calculated from the width
    and the number of horizontal and vertical pixels. The default width and aspect
    ratio approximate a 35mm camera sensor.

    Each pixel targets a randomly sampled point within the lens circle, modeled as a thin lens.
    The lens radius is determined by the f-number and focal length.
    The total number of rays sampled per pixel is `per_pixel_samples` multiplied by `lens_samples`.

    Parameters
    ----------
    pixels : tuple
        Tuple specifying the pixel dimensions of the camera (default=(720, 480)).
    width : float
        The CCD sensor width in metres (default=35mm).
    focal_length : float
        The focal length in metres (default=10mm).
        This value should match the lens specification.
    working_distance : float
        The distance from the lens to the focus plane in metres (default=50cm).
    ccd_distance : float, optional
        The distance between the CCD sensor and the lens (default: calculated from working_distance).
        If specified, `working_distance` is recalculated.
    f_number : float
        The f-number of the lens (default=3.5).
    lens_samples : int
        Number of samples to generate on the thin lens (default=100).
    per_pixel_samples : int
        Number of samples to generate per pixel (default=10).
        The total number of rays per pixel is `per_pixel_samples` x `lens_samples`.
    pipelines : list
        List of pipelines to process the spectrum measured at each pixel (default: [RGBPipeline2D()]).
    **kwargs : dict, optional
        Additional properties for the observer, such as parent, transform, pipelines, etc.
    """

    def __init__(
        self,
        pixels: tuple[int, int] = (720, 480),
        width: float = 0.035,
        focal_length: float = 10.0e-3,
        working_distance: float = 50.0e-2,
        ccd_distance: float | None = None,
        f_number: float = 3.5,
        lens_samples: int = 100,
        per_pixel_samples: int = 10,
        parent: _NodeBase | None = None,
        transform: AffineMatrix3D | None = None,
        name: str | None = None,
        pipelines: list[Pipeline2D] | None = None,
    ): ...
    @property
    def pixels(self) -> tuple[int, int]:
        """Tuple describing the pixel dimensions (nx, ny), e.g., (512, 512)."""

    @pixels.setter
    def pixels(self, value: tuple[int, int]) -> None: ...
    @property
    def width(self) -> float:
        """The CCD sensor width in metres."""

    @width.setter
    def width(self, value: float) -> None: ...
    @property
    def pixel_area(self) -> float:
        """Area of a single pixel on the CCD sensor."""

    @property
    def focal_length(self) -> float:
        """The focal length in metres."""

    @focal_length.setter
    def focal_length(self, value: float) -> None: ...
    @property
    def f_number(self) -> float:
        """The f-number, which defines the lens radius with the focal length."""

    @f_number.setter
    def f_number(self, value: float) -> None: ...
    @property
    def working_distance(self) -> float:
        """Distance from the lens to the focus plane in metres."""

    @working_distance.setter
    def working_distance(self, value: float) -> None: ...
    @property
    def ccd_distance(self) -> float:
        """Distance between the CCD sensor and the lens in metres."""

    @ccd_distance.setter
    def ccd_distance(self, value: float) -> None: ...
    @property
    def lens_radius(self) -> float:
        """Radius of the thin lens in metres, derived from focal length and f-number."""

    @property
    def lens_samples(self) -> int:
        """Number of samples to generate on the thin lens."""

    @lens_samples.setter
    def lens_samples(self, value: int) -> None: ...
    @property
    def per_pixel_samples(self) -> int:
        """Number of samples to generate per pixel."""

    @per_pixel_samples.setter
    def per_pixel_samples(self, value: int) -> None: ...
