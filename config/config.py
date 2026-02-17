import os
import warnings
import numpy as np
from osgeo import gdal
import inspect
import re
import textwrap


class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass


class Configuration:
    """ Configuration of all parameters used for DEM/DSM creation """

    def __init__(self):

        # --------- RUN NAME ---------
        self.run_name = 'MyCoolTest'  # Custom name for this run

        # ---------- PATHS -----------
        # Input data paths
        self.target_area_dir = '/path/to/my/target_areas'  # Path to vector footprints of target areas
        self.las_files_dir = '/path/to/my/lidar/pointclouds'  # Path to lidar point clouds (*.las/*.laz)
        self.las_footprints_dir = '/path/to/my/lidar/footprints'  # Path to footprints of lidar swath, if not available will be generated

        # Output directories
        self.preprocessed_dir = '/path/to/my/lidar/preprocessed'  # Path for preprocessed lidar data
        self.results_dir = '/path/to/my/lidar/results'  # Path for final DEM/DSM results
        self.validation_dir = '/path/to/my/lidar/validation'  # Path to validation data

        # ------ PREPROCESSING ------

        self.multiple_targets = True  # If target areas are saved in one gdf set to True
        self.target_name_field = 'id'  # If multiple target areas are present, this field in the target area gdf will be used as the target name

        self.max_elevation_threshold = 0.99 # Quantile to disgard atmospheric noise etc. Data outside the quantile is disgarded. 

        # SOR parameters
        self.knn = 100  # Number of k nearest neighbors, the higher the more stable
        self.multiplier = 2 # Threshold for outlier removal: points beyond (global_mean + multiplier * stddev) are removed.

        # ------- PROCESSING --------

        self.create_DSM = True
        self.create_DEM = False
        self.create_CHM = False

        self.fill_gaps = True # Whether to use IDW to close gaps in rasters
        self.resolution = 1 # Resolution of generated rasters in meter, can be 'Auto' or number

        self.point_density_method = 'sampling' # Method to determine point density, can be 'sampling' (exact) or 'density' (fast)

    # ______ GROUND FILTERING ______

        self.smrf_filter = True # Use SMRF ground filtering method (good for rough terrain)
        self.csf_filter = True # Use cloth simulation method (good for most terrains)
        self.threshold = 2 # Distance threshold to classify ground points, the lower the more points are classified as ground

        self.smrf_window_size = 20 # Window size for SMRF filter, the higher the more vegetation is removed
        self.smrf_slope = 0.2 # Slope for SMRF filter, the higher the more vegetation is removed
        self.smrf_scalar = 2 # Scalar for SMRF filter, the higher the more vegetation is removed

        self.csf_rigidness = 3 # Rigidness of the simulated cloth, the lower the more flexible, use low values for steep and high for flat terrain
        self.csf_iterations = 500 # Number of simulation steps, the higher, the more adapted to the point cloud
        self.csf_time_step = 0.5 # Time step of the simulation, the lower the more accurate, but slower
        self.csf_cloth_resolution = 1 # Resolution of the cloth (m), the lower the more accurate, but slower

        # ------ VALIDATION ------

        self.data_type = 'raster'   # Type of validation data, can be 'raster' or 'vector' (points)
        self.validation_target = 'DSM' # Product to validate, can be 'DSM', 'DEM' or 'CHM', select validation data accordingly! (DSM: higest point, DEM: ground level, CHM: height of vegetation)
        self.val_column_point = 'val_value' # Column in point validation data to use for comparison
        self.val_band_raster = 1 # Band in raster validation data to use for comparison
        self.sample_size = 100 # Number of points to sample for validation


        # ------ ADVANCED SETTINGS ------

        # _______ Preprocessing _______
        self.overlap = 0.2  # Minimum overlap between pointcloud and AOI needed for matching, 0.5 means 50% overlap

        self.filter_date = False  # Filter las files by date
        self.start_date = '2023-07-22'  # Start date for filtering las files
        self.end_date = '2026-07-20'  # End date for filtering las files

        # _______ Processing _______
        self.chunk_size = 500 # Chunk size in meters for parallel processing
        self.chunk_overlap = 0.1 # Overlap between chunks in percentage, 0.2 means 20% overlap
        self.num_workers = 16  # Number of parallel workers for processing

        # Set overall GDAL settings
        gdal.UseExceptions()  # Enable exceptions instead of silent failures
        gdal.SetCacheMax(32000000000)  # Set cache size in KB for GDAL operations
        warnings.filterwarnings('ignore')  # Suppress warnings

    def validate(self):
        """Validate config to catch errors early, and not during or at the end of processing"""

        # Check that required input paths exist
        for path_attr in ["target_area_dir", "las_footprints_dir", "las_files_dir"]:
            path = getattr(self, path_attr)
            if not os.path.exists(path):
                raise ConfigError(f"Invalid path: {path_attr} = {path}")

        # Create required output directories if they don't exist
        for path_attr in ["results_dir", "preprocessed_dir"]:
            path = getattr(self, path_attr)
            try:
                os.makedirs(path, exist_ok=True)
            except OSError:
                raise ConfigError(f"Unable to create folder: {path_attr} = {path}")

        return self


# ---- helper to expose inline comments as help text ----
def get_help_map() -> dict[str, str]:
    """
    Parse trailing '# ...' comments from assignments in Configuration.__init__.
    Returns { variable_name: "comment text" }.
    """
    try:
        src = inspect.getsource(Configuration.__init__)
    except Exception:
        return {}
    help_map: dict[str, str] = {}
    for line in textwrap.dedent(src).splitlines():
        m = re.match(r'\s*self\.(\w+)\s*=\s*.+?#\s*(.+)$', line)
        if m:
            var, comment = m.group(1), m.group(2).strip()
            help_map[var] = comment
    return help_map
