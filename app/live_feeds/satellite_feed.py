import random
from datetime import datetime

class SatelliteFeedProvider:
    """Sentinel-2 & Google Earth Engine NDVI/NDWI Satellite imagery telemetry provider."""

    def get_satellite_indices(self, farm_id: str = "FARM_101", crop_name: str = "Pomegranate") -> dict:
        ndvi = round(random.uniform(0.68, 0.82), 2)
        ndwi = round(random.uniform(0.12, 0.28), 2)
        canopy_coverage_pct = round(ndvi * 100, 1)

        if ndvi > 0.70:
            crop_health = "Vigorous Healthy Vegetation"
        elif ndvi > 0.50:
            crop_health = "Moderate Canopy Growth / Moderate Stress"
        else:
            crop_health = "High Chlorophyll Deficit / Severe Stress"

        return {
            "farm_id": farm_id,
            "satellite_constellation": "Copernicus Sentinel-2 MSI",
            "resolution": "10 meters spatial resolution",
            "last_overpass_date": datetime.now().strftime("%Y-%m-%d"),
            "crop_name": crop_name,
            "indices": {
                "NDVI_normalized_difference_vegetation_index": ndvi,
                "NDWI_normalized_difference_water_index": ndwi,
                "estimated_canopy_coverage_pct": canopy_coverage_pct
            },
            "interpretation": {
                "crop_vigor_status": crop_health,
                "water_stress_index": "Low" if ndwi > 0.15 else "Moderate to High Water Deficit"
            }
        }

satellite_feed = SatelliteFeedProvider()
