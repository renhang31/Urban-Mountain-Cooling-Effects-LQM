import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import Point, box
from scipy.signal import savgol_filter, argrelextrema
import matplotlib.pyplot as plt
import pandas as pd
"""
========================================================
Cooling Metrics Extraction Script (PCD, PCI, PCA)
Fully Reproducible Version — All Paths Relative to Script

This script calculates the park cooling distance (PCD),
cooling intensity (PCI), and cooling buffer temperature
profiles using LST data and a park boundary polygon.

All input and output paths are automatically adapted to
the directory where this script is located, ensuring full
portability and reproducibility for scientific workflow.
========================================================
"""
# ----------------------------------------------------------------------
# Global configuration
# ----------------------------------------------------------------------
TARGET_CRS = "EPSG:32648"      # UTM Zone 48N (suitable for Chengdu)
BUFFER_RADII = np.arange(30, 1201, 30)   # Radii of buffer rings (meters)

# Get the script directory as the universal base path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define standardized relative paths
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Ensure folders exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def reproject_geographic_raster(src_path, dst_crs=TARGET_CRS):
    """Reproject geographic-coordinate rasters to the target CRS."""
    with rasterio.open(src_path) as src:
        if src.crs.is_geographic:
            from rasterio.warp import calculate_default_transform, reproject, Resampling

            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )

            profile = src.profile
            profile.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            temp_path = os.path.join(os.path.dirname(src_path), "temp_reproj.tif")
            with rasterio.open(temp_path, 'w', **profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear
                    )
            return temp_path
        else:
            return src_path


def load_data(park_shp_path, lst_tif_path):
    """Load and harmonize input data under a unified CRS."""
    reproj_tif = reproject_geographic_raster(lst_tif_path)

    with rasterio.open(reproj_tif) as src:
        lst = src.read(1)
        transform = src.transform
        crs = src.crs
        raster_bounds = src.bounds

    park = gpd.read_file(park_shp_path).to_crs(crs)
    if park.empty:
        raise ValueError("Park polygon is empty or CRS conversion failed.")

    centroid = park.geometry.centroid.iloc[0]
    raster_box = box(*raster_bounds)

    if not raster_box.contains(centroid):
        raise ValueError("Park centroid lies outside the LST raster extent.")

    return park, lst, transform, crs, gpd.GeoSeries([centroid], crs=crs)


def generate_buffers(center_series):
    """Generate ring buffers (in meters) around the park centroid."""
    c = center_series.iloc[0]
    return gpd.GeoSeries([c.buffer(r).exterior.buffer(30) for r in BUFFER_RADII],
                         crs=center_series.crs)


def safe_extract_temperature(buffer, lst_data, transform):
    """Extract mean LST from each buffer, with minimum valid-pixel threshold."""
    mask = features.geometry_mask([buffer], lst_data.shape, transform, invert=True)
    masked = np.where(mask, lst_data, np.nan)

    valid_count = np.count_nonzero(~np.isnan(masked))
    if valid_count < 10:
        return np.nan
    return np.nanmean(masked)


def detect_cooling_edge(centroid_series, lst_data, transform):
    """Extract raw LST profile and detect temperature inflection point."""
    buffers = generate_buffers(centroid_series)
    temps = [safe_extract_temperature(buf, lst_data, transform) for buf in buffers]

    valid_mask = ~np.isnan(temps)
    if np.sum(valid_mask) < 5:
        raise ValueError("Insufficient valid buffer zones for inflection detection.")

    vt = np.array(temps)[valid_mask]
    smoothed = savgol_filter(vt, 5, 2)
    grad = np.gradient(smoothed)
    minima = argrelextrema(grad, np.less)[0]

    return temps, smoothed, minima, valid_mask


def plot_cooling_curve(radii, temps, smooth, minima, valid_mask, output_path):
    """Plot distance–temperature profile and mark the inflection point."""
    plt.figure(figsize=(10, 6))
    plt.plot(radii[valid_mask], np.array(temps)[valid_mask],
             'bo--', label='Raw Temperature', markersize=5)
    plt.plot(radii[valid_mask], smooth, 'r-', linewidth=2, label='Smoothed Curve')

    if len(minima) > 0:
        inf_idx = minima[0]
        plt.scatter(radii[valid_mask][inf_idx], smooth[inf_idx],
                    s=120, c='g', zorder=5, label="Inflection Point")

    plt.xlabel("Distance from Park Boundary (m)")
    plt.ylabel("Land Surface Temperature (°C)")
    plt.title("Cooling Profile")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def process_single_file(park_path, lst_path, output_dir):
    """Main processing function."""
    park, lst, transform, crs, centroid = load_data(park_path, lst_path)

    raw, smooth, minima, valid_mask = detect_cooling_edge(centroid, lst, transform)

    base = os.path.splitext(os.path.basename(lst_path))[0]

    # Save curve
    plot_path = os.path.join(output_dir, f"{base}_cooling_curve.png")
    plot_cooling_curve(BUFFER_RADII, raw, smooth, minima, valid_mask, plot_path)

    # Save CSV
    df = pd.DataFrame({
        "Distance_m": BUFFER_RADII[valid_mask],
        "Raw_LST": np.array(raw)[valid_mask],
        "Smoothed_LST": smooth
    })
    df.to_csv(os.path.join(output_dir, f"{base}_cooling_profile.csv"),
              index=False, encoding="utf-8-sig")

    result = {
        "File": base,
        "PCD_m": BUFFER_RADII[valid_mask][minima[0]] if len(minima) else np.nan,
        "PCI_degC": smooth[minima[0]] - smooth[0] if len(minima) else np.nan
    }
    return result


if __name__ == "__main__":
    # Default relative input/output paths
    park_shp = os.path.join(INPUT_DIR, "park_boundary.shp")
    lst_tif = os.path.join(INPUT_DIR, "LST.tif")

    result = process_single_file(park_shp, lst_tif, OUTPUT_DIR)

    pd.DataFrame([result]).to_csv(os.path.join(OUTPUT_DIR, "cooling_result.csv"),
                                  index=False)

    print("Processing completed. Result:")
    print(result)
