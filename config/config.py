import os
import warnings
import numpy as np
from osgeo import gdal


class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass


class Configuration:
    """ Configuration of all parameters used for DEM/DSM creation """

    def __init__(self):

        # --------- RUN NAME ---------
        self.run_name = 'Perma-X-2025-Ingmar'  # Custom name for this run
        self.year = 2025

        # ---------- PATHS -----------
        # Input data paths
        self.target_area_dir = '/isipd/projects/p_planetdw/data/lidar/01_target_areas/Ingmar'  # Path to vector footprints of target areas
        self.las_files_dir = '/isipd/projects-noreplica/p_macsprocessing/PermaX_MACS/PermaX_2025/data_products/WC_PeelSlumps_20250801_15cm_03/PointClouds'  # Path to lidar point clouds (*.las/*.laz)
        self.las_footprints_dir = f'/isipd/projects/p_planetdw/data/lidar/03_las_footprints/Ingmar'  # Path to footprints of flight paths, if not available will be generated

        # Output directories
        self.preprocessed_dir = '/isipd/projects/p_planetdw/data/lidar/04_preprocessed'  # Path for preprocessed lidar data
        self.results_dir = '/isipd/projects/p_planetdw/data/lidar/05_results'  # Path for final DEM/DSM results
        self.validation_dir = '/isipd/projects/p_planetdw/data/lidar/06_validation'  # Path to validation data

        # ------ PREPROCESSING ------

        self.multiple_targets = False  # If target areas are saved in one gdf set to True
        self.target_name_field = 'fid'  # Field in target area gdf to use as target name

        # elevation outlier cap (quantile in [0–1])
        self.max_elevation_threshold = 0.99  # Higher removes more high outliers (aircraft/atmosphere). Typical 0.98–0.9995.

        # SOR parameters (Statistical Outlier Removal)
        self.knn = 100  # neighbors for stats. 50–200 is common. Higher = stabler but slower.
        self.multiplier = 1  # (mean + m*std). Lower (0.8–1.2) = aggressive; higher (1.5–2.5) = conservative.

        # ------- PROCESSING --------

        self.create_DSM = False
        self.create_DEM = True
        self.create_CHM = False

        self.fill_gaps = True  # use IDW to close gaps in rasters
        self.resolution = 1  # pixel size (m). Smaller = sharper/heavier. Rule of thumb: >= sqrt(1 / points_per_m²).

        self.point_density_method = 'density'  # method to determine point density, can be 'sampling' (exact) or 'density' (fast)

        # ______ GROUND FILTERING ______

        self.smrf_filter = True  # use SMRF filter 
        self.csf_filter = True  # use cloth simulation method
        self.threshold = 0.5  # vertical tolerance (m) for extra clipping. Typical 0.5–2.

        # SMRF
        self.smrf_window_size = 20  # window size (m). Larger (15–30) removes more canopy but may bridge narrow valleys.
        self.smrf_slope = 0.2  # slope tolerance. Higher (0.2–0.4) accepts steeper ground; too high may keep low veg.
        self.smrf_scalar = 2  # elevation diff scale. 1–3 typical. Higher = more aggressive ground acceptance.

        # CSF (Cloth Simulation)
        self.csf_rigidness = 3  # cloth stiffness. 1–2 for rugged/steep; 3–4 for very flat urban.
        self.csf_iterations = 500  # steps. 200–1000. More = better fit, slower.
        self.csf_time_step = 1  # integration step. 0.5–1.0 common. Smaller = stable/accurate, slower.
        self.csf_cloth_resolution = 1  # grid spacing (m). 0.5–2 typical. Smaller = finer ground detail, heavier.

        # ------ VALIDATION ------

        self.data_type = 'raster'   # Type of validation data, can be 'raster' or 'vector' (points)
        self.validation_target = 'DSM' # product to validate, can be 'DSM', 'DEM' or 'CHM', select validation data accordingly! (DSM: higest point, DEM: ground level, CHM: height of vegetation)
        self.val_column_point = 'val_value' # column in point validation data to use for comparison
        self.val_band_raster = 1  # band index of reference raster (usually 1 unless multiband).
        self.sample_size = 100  # samples for stats. 100–10,000. Larger = more stable metrics.

        # ------ ADVANCED SETTINGS ------

        # _______ Preprocessing _______
        self.overlap = 0.0  # min overlap (fraction) between pointcloud and AOI. Typical 0.05–0.3.

        self.filter_date = False  # Filter las files by date
<<<<<<< HEAD

        self.automatic_date_parser = True # get dates from target area only for Region_Site_Date_Res_Order filenames


        self.start_date = '2023-07-10'  # Start date for filtering las files
        self.end_date = '2023-07-10'  # End date for filtering las files
=======
        self.start_date = '2023-07-22'  # Start date for filtering las files
        self.end_date = '2026-07-20'  # End date for filtering las files
>>>>>>> 3395522 (Update)

        # _______ Processing _______
        self.chunk_size = 1000  # chunk size (m). 250–1000 typical. Larger = fewer edges, more memory.
        self.chunk_overlap = 0.1  # chunk overlap (fraction). 0.05–0.3. More reduces seam artifacts.
        self.num_workers = 8  # parallel workers. <= physical cores/RAM capacity.

        # _______ Strip-level ICP alignment _______
        self.enable_strip_icp = False  # Enable incremental ICP on strip-level ground merged LAZ before AOI-level merge
        self.icp_voxel_size = 1.0  # Voxel size (m) for ICP downsampling
        self.icp_max_corr_dist = 2.0  # Maximum correspondence distance (m) for ICP
        self.icp_max_iters = 80  # Maximum ICP iterations
        self.icp_overlap_buffer_m = 10.0  # Buffer around overlap bbox (m) used for ICP subsets
        self.icp_min_fitness = 0.30  # Minimum ICP fitness to accept transform
        self.icp_max_abs_dz = 1.00  # Maximum absolute dz shift (m) to accept transform
        self.icp_max_rotation_deg = 1.00  # Maximum rotation angle (degrees) to accept transform
        self.icp_fail_policy = "skip_merge"  # One of: skip_merge, merge_without_icp, skip_strip, raise
        self.icp_use_overlap_crop = True  # If True, run ICP on overlap-only cropped subsets
        self.debug_mode = False  # If True, fail-fast behaviour for ICP precheck/runtime errors
        self.strict_crs_check = True  # If True, require compatible CRS/units before ICP runs
        self.debug_artefacts = True  # If True, keep overlap subset artefacts for inspection
        self.min_points = 500  # Minimum points required in each overlap crop to run ICP
        self.max_pre_dxy = 100.0  # Max centroid XY distance (m) allowed before ICP
        self.max_pre_dz = 5.0  # Max median Z delta (m) allowed before ICP
        self.icp_dz_consistency_threshold = 1.5  # Max allowed difference between pre-check dz median and ICP dz (m)

        # Set overall GDAL settings
        gdal.UseExceptions()  # Enable exceptions instead of silent failures
        gdal.SetCacheMax(32000000000)  # GDAL cache bytes (~32 GB). Set to ~20–60% of available RAM for big rasters.
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
