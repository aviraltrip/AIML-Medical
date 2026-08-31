import pytest
from pulsepoint_ai.engines.connect.care_locator import (
    find_care,
    haversine_km,
    map_icd10_to_specialties,
)
from pulsepoint_ai.core.schemas.care import CareLocatorRequest
from pulsepoint_ai.core.schemas.common import SeverityTier


def test_haversine_distance_calculation():
    # Distance between two identical coordinates must be 0
    d0 = haversine_km(12.9716, 77.5946, 12.9716, 77.5946)
    assert round(d0, 4) == 0.0

    # Distance between Bangalore and Mysore (~128-145 km)
    d_mysore = haversine_km(12.9716, 77.5946, 12.2958, 76.6394)
    assert 120.0 <= d_mysore <= 160.0


def test_map_icd10_to_specialties():
    # E11 (Type 2 diabetes mellitus)
    specs = map_icd10_to_specialties(["E11.9"])
    assert len(specs) > 0
    assert isinstance(specs[0], str)

    # Empty list should return default specialty
    default_specs = map_icd10_to_specialties([])
    assert len(default_specs) == 1


def test_locate_care_contract():
    req = CareLocatorRequest(
        patient_lat=12.9716,
        patient_lon=77.5946,
        icd10_codes=["E11.9", "I10"],
        severity_tier=SeverityTier.HIGH,
        radius_km=50.0,
    )
    res = find_care(req)
    assert res.request_id is not None
    assert isinstance(res.doctors, list)
    assert isinstance(res.required_specialties, list)
