"""Test module for ThinLensCCDArray observer."""

from __future__ import annotations

import pytest
from numpy.testing import assert_allclose
from raysect.core import Point3D, Vector3D, rotate_y, translate
from raysect.optical import Ray
from raysect.optical.observer.pipeline import RGBPipeline2D

from cherab.nagdis.observers.thin_lens_ccd import ThinLensCCDArray


class TestThinLensCCDArray:
    """Test cases for ThinLensCCDArray observer."""

    def test_default_initialization(self):
        """Test default initialization parameters."""
        # Use parameters that result in a valid thin lens setup
        # working_distance must be > focal_length for positive CCD distance
        # Default focal_length is 0.01m, so working_distance=1.0m is valid
        observer = ThinLensCCDArray(focal_length=0.01, working_distance=1.0)

        assert observer.pixels == (720, 480)
        assert observer.width == 0.035
        assert observer.focal_length == 0.01
        assert observer.f_number == 3.5
        assert observer.lens_samples == 100
        assert observer.per_pixel_samples == 10
        assert observer.pixel_samples == 1000  # lens_samples * per_pixel_samples
        assert observer.working_distance == 1.0

    @pytest.mark.parametrize(
        "pixels",
        [
            (512, 512),
            (720, 480),
            (1920, 1080),
            (100, 100),
        ],
    )
    def test_pixels_property(self, pixels):
        """Test pixels property with various resolutions."""
        observer = ThinLensCCDArray(pixels=pixels, focal_length=0.01, working_distance=1.0)
        assert observer.pixels == pixels

    @pytest.mark.parametrize(
        "invalid_pixels",
        [
            (0, 512),
            (512, 0),
            (-10, 512),
            (512, -10),
            (512,),  # single value
            (512, 512, 512),  # three values
        ],
    )
    def test_invalid_pixels(self, invalid_pixels):
        """Test invalid pixels parameter raises appropriate errors."""
        with pytest.raises(ValueError):
            ThinLensCCDArray(pixels=invalid_pixels, focal_length=0.01, working_distance=1.0)

    @pytest.mark.parametrize(
        "width",
        [
            0.024,  # APS-C sensor
            0.035,  # 35mm sensor
            0.048,  # Medium format
            0.001,  # Very small sensor
        ],
    )
    def test_width_property(self, width):
        """Test width property with various sensor sizes."""
        observer = ThinLensCCDArray(width=width, focal_length=0.01, working_distance=1.0)
        assert observer.width == width
        # Check pixel area is calculated correctly
        pixel_width = width / observer.pixels[0]
        expected_pixel_area = pixel_width**2
        assert_allclose(observer.pixel_area, expected_pixel_area)

    @pytest.mark.parametrize("invalid_width", [0, -0.01, -1.0])
    def test_invalid_width(self, invalid_width):
        """Test invalid width values raise ValueError."""
        with pytest.raises(ValueError):
            ThinLensCCDArray(width=invalid_width, focal_length=0.01, working_distance=1.0)

    @pytest.mark.parametrize(
        "focal_length",
        [
            0.005,  # 5mm
            0.01,  # 10mm
            0.05,  # 50mm
            0.1,  # 100mm
            0.2,  # 200mm
        ],
    )
    def test_focal_length_property(self, focal_length):
        """Test focal length property with various values."""
        # Use appropriate working distance for focal length
        # working_distance must be > focal_length for positive CCD distance
        working_distance = max(focal_length * 2.1, 1.0)  # Ensure working_distance > focal_length
        observer = ThinLensCCDArray(focal_length=focal_length, working_distance=working_distance)
        assert observer.focal_length == focal_length
        # Check lens radius is calculated correctly
        expected_lens_radius = 0.5 * focal_length / observer.f_number
        assert_allclose(observer.lens_radius, expected_lens_radius)

    @pytest.mark.parametrize("invalid_focal_length", [0, -0.01, -1.0])
    def test_invalid_focal_length(self, invalid_focal_length):
        """Test invalid focal length values raise ValueError."""
        with pytest.raises(ValueError):
            ThinLensCCDArray(focal_length=invalid_focal_length, working_distance=1.0)

    @pytest.mark.parametrize("f_number", [1.0, 1.4, 2.8, 3.5, 5.6, 8.0, 11.0, 16.0])
    def test_f_number_property(self, f_number):
        """Test f-number property with common photography values."""
        observer = ThinLensCCDArray(f_number=f_number, focal_length=0.01, working_distance=1.0)
        assert observer.f_number == f_number
        # Check lens radius is calculated correctly
        expected_lens_radius = 0.5 * observer.focal_length / f_number
        assert_allclose(observer.lens_radius, expected_lens_radius)

    @pytest.mark.parametrize("invalid_f_number", [0, -1.0, -3.5])
    def test_invalid_f_number(self, invalid_f_number):
        """Test invalid f-number values raise ValueError."""
        with pytest.raises(ValueError):
            ThinLensCCDArray(f_number=invalid_f_number, focal_length=0.01, working_distance=1.0)

    @pytest.mark.parametrize(
        "focal_length,working_distance",
        [
            (0.005, 0.1),  # 5mm lens, 10cm working distance
            (0.01, 0.5),  # 10mm lens, 50cm working distance
            (0.05, 1.0),  # 50mm lens, 1m working distance
            (0.1, 2.0),  # 100mm lens, 2m working distance
            (0.2, 10.0),  # 200mm lens, 10m working distance
        ],
    )
    def test_working_distance_property(self, focal_length, working_distance):
        """Test working distance property with various values."""
        observer = ThinLensCCDArray(focal_length=focal_length, working_distance=working_distance)
        assert observer.working_distance == working_distance
        # Check CCD distance is calculated correctly using thin lens equation
        expected_ccd_distance = 1 / (1 / focal_length - 1 / working_distance)
        assert_allclose(observer.ccd_distance, expected_ccd_distance, rtol=1e-10)

    @pytest.mark.parametrize("invalid_working_distance", [0, -0.1, -1.0])
    def test_invalid_working_distance(self, invalid_working_distance):
        """Test invalid working distance values raise ValueError."""
        with pytest.raises(ValueError):
            ThinLensCCDArray(working_distance=invalid_working_distance, focal_length=0.01)

    @pytest.mark.parametrize(
        "focal_length,ccd_distance",
        [
            (0.01, 0.011),  # 10mm lens, 11mm CCD distance
            (0.05, 0.055),  # 50mm lens, 55mm CCD distance
            (0.1, 0.11),  # 100mm lens, 110mm CCD distance
        ],
    )
    def test_ccd_distance_property(self, focal_length, ccd_distance):
        """Test CCD distance property when explicitly set."""
        observer = ThinLensCCDArray(focal_length=focal_length, ccd_distance=ccd_distance)
        assert observer.ccd_distance == ccd_distance
        # Check working distance is recalculated using thin lens equation
        expected_working_distance = 1 / (1 / focal_length - 1 / ccd_distance)
        assert_allclose(observer.working_distance, expected_working_distance, rtol=1e-10)

    @pytest.mark.parametrize("invalid_ccd_distance", [0, -0.01, -1.0])
    def test_invalid_ccd_distance(self, invalid_ccd_distance):
        """Test invalid CCD distance values raise ValueError."""
        with pytest.raises(ValueError):
            ThinLensCCDArray(ccd_distance=invalid_ccd_distance, focal_length=0.01)

    @pytest.mark.parametrize("lens_samples", [10, 50, 100, 200, 500])
    def test_lens_samples_property(self, lens_samples):
        """Test lens samples property with various values."""
        observer = ThinLensCCDArray(
            lens_samples=lens_samples, focal_length=0.01, working_distance=1.0
        )
        assert observer.lens_samples == lens_samples
        # Check total pixel samples is updated
        expected_total_samples = lens_samples * observer.per_pixel_samples
        assert observer.pixel_samples == expected_total_samples

    @pytest.mark.parametrize("invalid_lens_samples", [0, -1, -10])
    def test_invalid_lens_samples(self, invalid_lens_samples):
        """Test invalid lens samples values raise ValueError."""
        with pytest.raises(ValueError):
            ThinLensCCDArray(
                lens_samples=invalid_lens_samples, focal_length=0.01, working_distance=1.0
            )

    @pytest.mark.parametrize("per_pixel_samples", [1, 5, 10, 20, 50])
    def test_per_pixel_samples_property(self, per_pixel_samples):
        """Test per pixel samples property with various values."""
        observer = ThinLensCCDArray(
            per_pixel_samples=per_pixel_samples, focal_length=0.01, working_distance=1.0
        )
        assert observer.per_pixel_samples == per_pixel_samples
        # Check total pixel samples is updated
        expected_total_samples = observer.lens_samples * per_pixel_samples
        assert observer.pixel_samples == expected_total_samples

    @pytest.mark.parametrize("invalid_per_pixel_samples", [0, -1, -10])
    def test_invalid_per_pixel_samples(self, invalid_per_pixel_samples):
        """Test invalid per pixel samples values raise ValueError."""
        with pytest.raises(ValueError):
            ThinLensCCDArray(
                per_pixel_samples=invalid_per_pixel_samples, focal_length=0.01, working_distance=1.0
            )

    def test_thin_lens_equation_consistency(self):
        """Test that thin lens equation is consistently applied."""
        focal_length = 0.05  # 50mm
        working_distance = 1.0  # 1m

        # Test setting working distance first
        observer1 = ThinLensCCDArray(focal_length=focal_length, working_distance=working_distance)

        # Test setting CCD distance directly
        ccd_distance = 1 / (1 / focal_length - 1 / working_distance)
        observer2 = ThinLensCCDArray(focal_length=focal_length, ccd_distance=ccd_distance)

        # Both should give the same result
        assert_allclose(observer1.working_distance, observer2.working_distance, rtol=1e-10)
        assert_allclose(observer1.ccd_distance, observer2.ccd_distance, rtol=1e-10)

    @pytest.mark.parametrize(
        "pixels,width,focal_length,f_number,working_distance",
        [
            ((512, 512), 0.024, 0.05, 2.8, 2.5),
            ((720, 480), 0.035, 0.085, 1.8, 4.25),
            ((1920, 1080), 0.036, 0.024, 5.6, 1.2),
        ],
    )
    def test_geometry_calculations(self, pixels, width, focal_length, f_number, working_distance):
        """Test geometric calculations for various camera configurations."""
        observer = ThinLensCCDArray(
            pixels=pixels,
            width=width,
            focal_length=focal_length,
            f_number=f_number,
            working_distance=working_distance,
        )

        # Test pixel area calculation
        pixel_width = width / pixels[0]
        expected_pixel_area = pixel_width**2
        assert_allclose(observer.pixel_area, expected_pixel_area)

        # Test lens radius calculation
        expected_lens_radius = 0.5 * focal_length / f_number
        assert_allclose(observer.lens_radius, expected_lens_radius)

    def test_ray_generation(self):
        """Test ray generation functionality."""
        observer = ThinLensCCDArray(
            pixels=(10, 10),
            lens_samples=5,
            per_pixel_samples=2,
            focal_length=0.01,
            working_distance=1.0,
        )

        # Create a template ray
        template_ray = Ray(origin=Point3D(0, 0, 0), direction=Vector3D(0, 0, 1))

        # Generate rays for a pixel
        rays = observer._generate_rays(5, 5, template_ray, 10)

        # Should generate lens_samples * per_pixel_samples rays
        expected_ray_count = observer.lens_samples * observer.per_pixel_samples
        assert len(rays) == expected_ray_count

        # Each element should be a tuple of (ray, weight)
        for ray_tuple in rays:
            assert len(ray_tuple) == 2
            ray, weight = ray_tuple
            assert isinstance(ray, Ray)
            assert isinstance(weight, float)
            assert weight > 0  # Weight should be positive

    def test_pixel_sensitivity(self):
        """Test pixel sensitivity calculation."""
        observer = ThinLensCCDArray(
            width=0.035, pixels=(720, 480), focal_length=0.01, working_distance=1.0
        )

        # Pixel sensitivity should be constant for all pixels
        sensitivity_center = observer._pixel_sensitivity(360, 240)
        sensitivity_corner = observer._pixel_sensitivity(0, 0)
        sensitivity_edge = observer._pixel_sensitivity(360, 0)

        # All should be equal for this simple implementation
        assert_allclose(sensitivity_center, sensitivity_corner)
        assert_allclose(sensitivity_center, sensitivity_edge)

        # Should be positive
        assert sensitivity_center > 0

    def test_custom_pipelines(self):
        """Test initialization with custom pipelines."""
        custom_pipeline = RGBPipeline2D()
        observer = ThinLensCCDArray(
            pipelines=[custom_pipeline], focal_length=0.01, working_distance=1.0
        )

        assert len(observer.pipelines) == 1
        assert observer.pipelines[0] == custom_pipeline

    def test_inheritance_properties(self):
        """Test that Observer2D properties are properly inherited."""
        observer = ThinLensCCDArray(
            parent=None,
            transform=translate(1, 2, 3) * rotate_y(45),
            name="test_camera",
            focal_length=0.01,
            working_distance=1.0,
        )

        assert observer.name == "test_camera"
        assert observer.parent is None
        # Transform should be set
        assert observer.transform is not None
        assert observer.pixel_samples == observer.lens_samples * observer.per_pixel_samples

    @pytest.mark.parametrize(
        "focal_length,f_number,working_distance,expected_depth_of_field_factor",
        [
            (0.05, 1.4, 2.5, "large"),  # Large aperture, shallow DoF
            (0.05, 8.0, 2.5, "small"),  # Small aperture, deep DoF
        ],
    )
    def test_depth_of_field_behavior(
        self, focal_length, f_number, working_distance, expected_depth_of_field_factor
    ):
        """Test that lens parameters affect depth of field as expected."""
        observer = ThinLensCCDArray(
            focal_length=focal_length, f_number=f_number, working_distance=working_distance
        )

        # Larger aperture (smaller f-number) should give larger lens radius
        lens_radius = observer.lens_radius
        expected_lens_radius = 0.5 * focal_length / f_number
        assert_allclose(lens_radius, expected_lens_radius)

        if expected_depth_of_field_factor == "large":
            # Large aperture should give larger lens radius
            assert lens_radius > 0.01
        else:
            # Small aperture should give smaller lens radius
            assert lens_radius < 0.01

    def test_realistic_camera_setup(self):
        """Test a realistic camera setup similar to actual experiments."""
        # Typical scientific camera setup
        observer = ThinLensCCDArray(
            pixels=(2048, 2048),  # High-resolution scientific camera
            width=0.014,  # 14mm sensor
            focal_length=0.025,  # 25mm lens
            f_number=4.0,
            working_distance=0.3,  # 30cm working distance
            lens_samples=50,
            per_pixel_samples=20,
        )

        # Verify setup makes physical sense
        assert observer.ccd_distance > 0
        assert observer.lens_radius > 0
        assert observer.pixel_area > 0
        assert observer.pixel_samples == 1000  # 50 * 20

        # Test lens equation
        f = observer.focal_length
        d_o = observer.working_distance
        d_i = observer.ccd_distance
        assert_allclose(1 / f, 1 / d_o + 1 / d_i, rtol=1e-10)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Very small focal length with appropriate working distance
        observer = ThinLensCCDArray(focal_length=0.001, working_distance=0.01)
        assert observer.ccd_distance > 0

        # Very large working distance
        observer = ThinLensCCDArray(focal_length=0.05, working_distance=100.0)
        assert_allclose(observer.ccd_distance, observer.focal_length, rtol=1e-3)

        # Minimum samples
        observer = ThinLensCCDArray(
            lens_samples=1, per_pixel_samples=1, focal_length=0.01, working_distance=1.0
        )
        assert observer.pixel_samples == 1

    def test_property_updates_affect_calculations(self):
        """Test that changing properties updates dependent calculations."""
        observer = ThinLensCCDArray(focal_length=0.01, working_distance=1.0)

        # Store initial values
        initial_lens_radius = observer.lens_radius
        initial_ccd_distance = observer.ccd_distance
        initial_pixel_area = observer.pixel_area

        # Change f-number
        observer.f_number = 1.4
        assert observer.lens_radius != initial_lens_radius

        # Change working distance
        observer.working_distance = 2.0
        assert observer.ccd_distance != initial_ccd_distance

        # Change width
        observer.width = 0.024
        assert observer.pixel_area != initial_pixel_area
