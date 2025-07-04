"""Module providing the ThinLensCCDArray observer object, which models a thin lens imaging sensor.
"""
from __future__ import annotations

from raysect.optical.observer.pipeline import RGBPipeline2D
from raysect.optical.observer.sampler2d import FullFrameSampler2D

cimport cython
from libc.math cimport M_PI
from raysect.core.math.sampler cimport DiskSampler3D, RectangleSampler3D
from raysect.optical cimport AffineMatrix3D, Point3D, Ray, Vector3D, translate
from raysect.optical.observer.base cimport Observer2D

__all__ = ["ThinLensCCDArray"]


cdef class ThinLensCCDArray(Observer2D):
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

    cdef:
        int _lens_samples, _per_pixel_samples
        double _width, _pixel_area, _image_delta, _image_start_x, _image_start_y
        double _focal_length, _working_distance, _f_number, _lens_radius, _lens_area
        double _ccd_distance
        RectangleSampler3D _pixel_sampler
        DiskSampler3D _lens_sampler

    def __init__(
        self,
        pixels=(720, 480),
        width=0.035,
        focal_length=10.e-3,
        working_distance=50.e-2,
        ccd_distance=None,
        f_number=3.5,
        lens_samples=100,
        per_pixel_samples=10,
        parent=None,
        transform=None,
        name=None,
        pipelines=None
    ):

        # Set initial values to avoid undefined behaviour when setting via setters
        self._width = 0.035
        self._pixels = (720, 480)
        self._focal_length = 10.e-3
        self._working_distance = 50.e-2
        self._ccd_distance = -1
        self._f_number = 3.5
        self._lens_samples = 100
        self._per_pixel_samples = 10

        pipelines = pipelines or [RGBPipeline2D()]

        super().__init__(pixels, FullFrameSampler2D(), pipelines,
                         pixel_samples=self._lens_samples * self._per_pixel_samples,
                         parent=parent, transform=transform, name=name)

        # Setting width and focal_length triggers calculation of image and lens geometry
        self.lens_samples = lens_samples
        self.per_pixel_samples = per_pixel_samples
        self.width = width
        self.f_number = f_number

        # Setting the following properties triggers calculation of a lens equation
        self.focal_length = focal_length
        self.working_distance = working_distance
        if ccd_distance is not None:
            self.ccd_distance = ccd_distance

    @property
    def pixels(self) -> tuple[int, int]:
        """Tuple describing the pixel dimensions (nx, ny), e.g., (512, 512).

        :rtype: tuple[int, int]
        """
        return self._pixels

    @pixels.setter
    def pixels(self, value):
        pixels = tuple(value)
        if len(pixels) != 2:
            raise ValueError("Pixels must be a 2-element tuple defining the x and y resolution.")
        x, y = pixels
        if x <= 0:
            raise ValueError("Number of x pixels must be greater than 0.")
        if y <= 0:
            raise ValueError("Number of y pixels must be greater than 0.")
        self._pixels = pixels
        self._update_image_geometry()

    @property
    def width(self) -> float:
        """The CCD sensor width in metres.

        :rtype: float
        """
        return self._width

    @width.setter
    def width(self, width):
        if width <= 0:
            raise ValueError("Width must be greater than 0 metres.")
        self._width = width
        self._update_image_geometry()

    @property
    def pixel_area(self) -> float:
        """Area of a single pixel on the CCD sensor.

        :rtype: float
        """
        return self._pixel_area

    @property
    def focal_length(self) -> float:
        """Focal length in metres.

        :rtype: float
        """
        return self._focal_length

    @focal_length.setter
    def focal_length(self, value):
        focal_length = value
        if focal_length <= 0:
            raise ValueError("Focal length must be greater than 0.")
        self._focal_length = focal_length
        self._update_lens_geometry()
        self._update_ccd_distance()

    @property
    def f_number(self) -> float:
        """The f-number, which defines the lens radius with the focal length.

        :rtype: float
        """
        return self._f_number

    @f_number.setter
    def f_number(self, value):
        f_number = value
        if f_number <= 0:
            raise ValueError("F-number must be greater than 0.")
        self._f_number = f_number
        self._update_lens_geometry()

    @property
    def working_distance(self) -> float:
        """Distance between the lens plane and the focus plane in metres.

        :rtype: float
        """
        return self._working_distance

    @working_distance.setter
    def working_distance(self, value):
        working_distance = value
        if working_distance <= 0:
            raise ValueError("Working distance must be greater than 0.")
        self._working_distance = working_distance
        self._update_lens_geometry()
        self._update_ccd_distance()

    @property
    def ccd_distance(self) -> float:
        """Distance between the CCD sensor and the lens in metres.

        :rtype: float
        """
        return self._ccd_distance

    @ccd_distance.setter
    def ccd_distance(self, value):
        ccd_distance = value
        if ccd_distance <= 0:
            raise ValueError("CCD distance must be greater than 0.")
        self._ccd_distance = ccd_distance
        self._update_lens_geometry()
        self._update_working_distance()

    @property
    def lens_radius(self) -> float:
        """Lens radius in metres.

        :rtype: float
        """
        return self._lens_radius

    @property
    def lens_samples(self) -> int:
        """Number of samples on the lens.

        :rtype: int
        """
        return self._lens_samples

    @lens_samples.setter
    def lens_samples(self, value):
        if value <= 0:
            raise ValueError("The number of lens samples must be greater than 0.")
        self._lens_samples = value
        self._update_pixel_samples()

    @property
    def per_pixel_samples(self) -> int:
        """Number of samples to take per pixel.

        :rtype: int
        """
        return self._per_pixel_samples

    @per_pixel_samples.setter
    def per_pixel_samples(self, value):
        if value <= 0:
            raise ValueError("The number of pixel samples must be greater than 0.")
        self._per_pixel_samples = value
        self._update_pixel_samples()

    cdef object _update_image_geometry(self):
        self._image_delta = self._width / self._pixels[0]
        self._image_start_x = 0.5 * self._pixels[0] * self._image_delta
        self._image_start_y = 0.5 * self._pixels[1] * self._image_delta
        self._pixel_sampler = RectangleSampler3D(self._image_delta, self._image_delta)
        self._pixel_area = (self._width / self._pixels[0])**2

    cdef object _update_lens_geometry(self):
        self._lens_radius = 0.5 * self._focal_length / self._f_number
        self._lens_sampler = DiskSampler3D(self._lens_radius)
        self._lens_area = M_PI * self._lens_radius**2

    cdef object _update_ccd_distance(self):
        self._ccd_distance = 1 / (1 / self._focal_length - 1 / self._working_distance)

    cdef object _update_working_distance(self):
        self._working_distance = 1 / (1 / self._focal_length - 1 / self._ccd_distance)

    cdef object _update_pixel_samples(self):
        # pixel_samples is the total number of samples per pixel
        self.pixel_samples = self._per_pixel_samples * self._lens_samples

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef list _generate_rays(self, int ix, int iy, Ray template, int ray_samples):
        """Generate a list of Rays that sample the sensitivity of a pixel.

        Returns a list of tuples, each containing a Ray object and its corresponding weight,
        typically the projected area/direction cosine.

        The weight is calculated as:

        .. math::

            weight := \\frac{R^2\\cos^4\\theta}{2d^2},

        where :math:`R` is the lens radius, :math:`\\theta` is the angle between the ray vector and
        the normal to the image sensor, and :math:`d` is the distance between the image sensor and
        the lens.
            :math:`R` and :math:`d` are calculated as:

        .. math::

                R &= \\frac{f}{F}

                d &= \\left(\\frac{1}{f} - \\frac{1}{W}\\right)^{-1}.

        where :math:`f` is the focal length, :math:`F` is the f-number, and :math:`W` is the
        working distance.

        Parameters
        ----------
        ix : int
            Pixel x index.
        iy : int
            Pixel y index.
        template : Ray
            The template ray from which all rays are generated.
        ray_samples : int
            The number of rays to generate (not used; determined by per_pixel_samples).

        Returns
        -------
        list[tuple[Ray, float]]
            A list of (ray, weight) tuples.
        """
        cdef:
            double pixel_x, pixel_y, weight
            list pixel_origins, lens_origins, rays
            Point3D pixel_origin, lens_origin
            Vector3D pixel_direction, direction
            AffineMatrix3D pixel_to_local

        # Compute pixel transform
        pixel_x = self._image_start_x - self._image_delta * ix
        pixel_y = self._image_start_y - self._image_delta * iy
        pixel_to_local = translate(pixel_x, pixel_y, -1 * self._ccd_distance)

        # Generate origin points in pixel space
        pixel_origins = self._pixel_sampler.samples(self._per_pixel_samples)

        # Assemble rays
        rays = []
        for pixel_origin in pixel_origins:

            # Transform to local space from pixel space
            pixel_origin = pixel_origin.transform(pixel_to_local)

            # Generate origin points in lens space (equal to local space)
            lens_origins = self._lens_sampler.samples(self._lens_samples)

            for lens_origin in lens_origins:

                # Generate direction from sampled pixel point to sampled lens point
                pixel_direction = pixel_origin.vector_to(lens_origin)
                pixel_direction = pixel_direction.normalise()

                # Generate ray direction from sampled lens point (lens_origin)
                direction = pixel_origin.vector_to(Point3D(0, 0, 0)).normalise()
                direction = lens_origin.vector_to(Point3D(0, 0, 0)) + direction * self._working_distance / direction.z
                direction = direction.normalise()

                # weight = 0.5 * lens_radius^2 * cos(theta)^4 / ccd_distance^2
                weight = 0.5 * self._lens_radius**2 * pixel_direction.z**4 / self._ccd_distance**2

                rays.append((template.copy(lens_origin, direction), weight))

        return rays

    cpdef double _pixel_sensitivity(self, int x, int y):
        return self._pixel_area * 2 * M_PI
