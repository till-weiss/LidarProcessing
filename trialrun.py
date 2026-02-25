import os
import time
from pathlib import Path
from datetime import timedelta

import laspy

import preprocessing as pre
import processing as pro   # instead of debug_processing as pro

import config.config as configuration

config = configuration.Configuration()
# --- EDIT THESE TO YOUR PATHS ---
config.run_name = "ICP_test"

config.target_area_dir = "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/data/area"
#target_files = glob.glob(os.path.join(config.target_area_dir, "*.gpkg"))

config.las_files_dir = "/isipd/projects/p_planetdw/data/lidar/02_pointclouds/2023"
config.las_footprints_dir = "/isipd/projects/p_planetdw/data/lidar/03_las_footprints/2023"

config.preprocessed_dir = "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/preprocessed"
config.results_dir = "/isipd/projects/Response/GIS_RS_projects/Masterarbeit_Till_Weiss/results"

# --- optional: first test run settings ---
config.num_workers = 16        
config.chunk_size = 500   
config.overlap = 0.2         

config.enable_strip_icp = True

config.create_DSM = True
config.create_DEM = True
config.create_CHM = False


def count_points(las_path: str) -> int:
    with laspy.open(las_path) as f:
        return int(f.header.point_count)

print("\n========== STEP A: Footprint matching (for logging) ==========")
t0 = time.time()

run_out = os.path.join(config.preprocessed_dir, config.run_name)
os.makedirs(run_out, exist_ok=True)

las_dict = pre.match_footprints(
    target_footprint_dir=config.target_area_dir,
    las_footprint_dir=config.las_footprints_dir,
    las_file_dir=config.las_files_dir,
    out_dir=os.path.join(config.preprocessed_dir, config.run_name),
    threshold=config.overlap,
    filter_date=config.filter_date,
    start_date=config.start_date,
    end_date=config.end_date
)

print(f"Footprint matching finished in {time.time() - t0:.1f}s")

print("\nTiles per target + raw point counts:")
raw_points_by_target = {}
for target, tiles in las_dict.items():
    raw_points = 0
    for tile in tiles:
        raw_points += count_points(tile)

    raw_points_by_target[target] = raw_points
    print(f"  {target}: tiles={len(tiles)} | raw_points={raw_points:,}")


print("\n========== STEP B: preprocess_all(config) ==========")
t1 = time.time()
pre.preprocess_all(config)
print(f"Preprocessing completed in {timedelta(seconds=int(time.time() - t1))}")

# Check cleaned outputs (raw → cleaned)
print("\nCleaned point counts (raw → cleaned):")
run_pre_dir = Path(config.preprocessed_dir) / config.run_name

for target, raw_points in raw_points_by_target.items():
    # expected output naming pattern from your earlier code:
    cleaned_las = run_pre_dir / f"{Path(target).stem}.las"
    if not cleaned_las.exists():
        print(f"  {Path(target).stem}: NOT FOUND -> {cleaned_las}")
        continue

    cleaned_points = count_points(str(cleaned_las))
    removed_pct = 100 * (1 - cleaned_points / raw_points) if raw_points else 0.0
    print(f"  {Path(target).stem}: {raw_points:,} → {cleaned_points:,} ({removed_pct:.1f}% removed)")


print("\n========== STEP C: process_all(config) ==========")
t2 = time.time()
pro.process_all(config)
print(f"Processing completed in {timedelta(seconds=int(time.time() - t2))}")

print("\nDone.")
print(f"Cleaned LAS: {Path(config.preprocessed_dir) / config.run_name}")
print(f"Results:     {Path(config.results_dir) / config.run_name}")