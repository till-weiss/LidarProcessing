import os
import time
import json
import shutil
from datetime import timedelta
from datetime import datetime
from multiprocessing import Pool

import pdal
import laspy
import numpy as np
from matplotlib import pyplot as plt
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import shape
from shapely.wkt import loads as wkt_loads, dumps as wkt_dumps

from core.reprojection import get_utm_epsg, reproject_las, is_utm_crs
from core.preprocess_windowed import create_chunks_from_wkt, process_chunk, merge_and_crop_chunks
from core.extract_footprints import extract_footprint_batch
from core.utils import split_gpkg

def _list_las_laz_files(las_file_dir):
    exts = (".las", ".laz")
    return sorted(
        os.path.join(las_file_dir, f)
        for f in os.listdir(las_file_dir)
        if not f.startswith(".") and f.lower().endswith(exts)
    )


def get_las_header(las_file):
    with laspy.open(las_file) as las:
        header = las.header
        scale = header.scales
        offset = header.offsets
        crs = header.parse_crs()
        crs_epsg = crs.to_epsg() if crs else 4979
    return scale, offset, crs_epsg


def process_chunk_wrapper(args):
    return process_chunk(*args)

def plot_target_and_footprints(target_gdf, matched_las_paths, las_footprint_dir, output_path):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot target area in red
    target_gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=2, label='Target Area')

    # Overlay LAS footprints in blue
    for las_path in matched_las_paths:
        las_name = os.path.splitext(os.path.basename(las_path))[0]
        las_fp_path = os.path.join(las_footprint_dir, las_name + ".gpkg")
        if os.path.exists(las_fp_path):
            las_gdf = gpd.read_file(las_fp_path)
            if las_gdf.crs != target_gdf.crs:
                las_gdf = las_gdf.to_crs(target_gdf.crs)
            las_gdf.plot(ax=ax, facecolor='blue', edgecolor='blue', alpha=0.3, label='Matched LAS Footprint')

    plt.title('Target Area and Matched LAS Footprints')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def match_footprints(
    target_footprint_dir,
    las_footprint_dir,
    las_file_dir,
    out_dir,
    threshold=0.5,
    filter_date=True,
    start_date=None,
    end_date=None,
    fallback_to_all_las=True,
):
    import os
    import time
    from datetime import datetime, timedelta

    import geopandas as gpd
    import laspy
    from tqdm import tqdm

    # ---- helper: list LAS/LAZ safely (ignores .DS_Store etc.) ----
    def _list_las_laz_files(folder):
        exts = (".las", ".laz")
        return sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if (not f.startswith(".")) and f.lower().endswith(exts)
        )

    os.makedirs(las_footprint_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    print("\nMatching Lidar footprints...")
    start = time.time()

    # Generate footprints if none exist
    if not any((f.endswith(".gpkg") and not f.startswith(".")) for f in os.listdir(las_footprint_dir)):
        print("No footprint files found. Generating footprints first.")
        extract_footprint_batch(las_file_dir, las_footprint_dir)

    target_footprints = [
        os.path.join(target_footprint_dir, f)
        for f in os.listdir(target_footprint_dir)
        if (not f.startswith(".")) and f.endswith(".gpkg")
    ]
    las_footprints = [
        os.path.join(las_footprint_dir, f)
        for f in os.listdir(las_footprint_dir)
        if (not f.startswith(".")) and f.endswith(".gpkg")
    ]

    # Fallback list (used only if matching returns 0 files for a target)
    all_las_laz = _list_las_laz_files(las_file_dir)

    target_dict = {}

    for target_fp in tqdm(target_footprints, desc="Finding target areas", unit="areas"):
        target_gdf = gpd.read_file(target_fp)
        target_name = os.path.splitext(os.path.basename(target_fp))[0]
        las_paths = []

        for las_fp in tqdm(las_footprints, desc="Checking LAS footprints", unit="footprints"):
            las_gdf = gpd.read_file(las_fp)

            # CRS harmonisation
            if target_gdf.crs != las_gdf.crs:
                las_gdf = las_gdf.to_crs(target_gdf.crs)

            # quick intersects check
            joined = gpd.sjoin(las_gdf, target_gdf, predicate="intersects")
            if joined.empty:
                continue

            # area overlap ratio
            intersection = gpd.overlay(las_gdf, target_gdf, how="intersection")
            intersection_area = float(intersection.area.sum())
            target_area = float(target_gdf.geometry.area.sum())

            if target_area <= 0:
                continue

            if (intersection_area / target_area) <= threshold:
                continue

            las_name = os.path.splitext(os.path.basename(las_fp))[0]

            # Check for both .las and .laz files
            las_path = os.path.join(las_file_dir, las_name + ".las")
            laz_path = os.path.join(las_file_dir, las_name + ".laz")

            if os.path.exists(las_path):
                chosen = las_path
            elif os.path.exists(laz_path):
                chosen = laz_path
            else:
                chosen = None

            if not chosen:
                continue

            # Optional date filter
            if filter_date and (start_date or end_date):
                sd = start_date
                ed = end_date

                if isinstance(sd, str):
                    sd = datetime.strptime(sd, "%Y-%m-%d").date()
                if isinstance(ed, str):
                    ed = datetime.strptime(ed, "%Y-%m-%d").date()

                try:
                    with laspy.open(chosen) as las_file:
                        las_date = las_file.header.creation_date

                    if not las_date:
                        continue
                    if sd and las_date < sd:
                        continue
                    if ed and las_date > ed:
                        continue
                except Exception as e:
                    print(f"Failed to read LAS header from {chosen}: {e}")
                    continue

            las_paths.append(chosen)

        # ✅ FALLBACK: if matching found nothing, proceed anyway
        if fallback_to_all_las and len(las_paths) == 0:
            print(
                f"Target area: {target_name} matched 0 footprints – "
                f"falling back to ALL LAS/LAZ in {las_file_dir}"
            )
            las_paths = all_las_laz.copy()

        target_dict[target_name] = las_paths

        # Optional plot for logging (only if you have this function)
        if las_paths:
            try:
                output_plot_path = os.path.join(out_dir, f"{target_name}_footprints.png")
                plot_target_and_footprints(target_gdf, las_paths, las_footprint_dir, output_plot_path)
            except Exception as e:
                print(f"Plotting footprints failed for {target_name}: {e}")

        print(f"Target area: {target_name}, LAS files found: {len(las_paths)}")

    print(
        f"Footprint matching completed in {timedelta(seconds=int(time.time() - start))}. "
        f"Found {len(target_dict)} target areas."
    )
    return target_dict


def merge_and_clean_las(las_dict, preprocessed_dir, run_name, target_footprint_dir, max_elev, sor_knn, sor_multiplier, num_workers, chunk_size=1000):

    run_merged_dir = os.path.join(preprocessed_dir, run_name)
    os.makedirs(run_merged_dir, exist_ok=True)

    print("\nProcessing LAS files in chunks...")
    start = time.time()

    for target_fp, las_files in tqdm(las_dict.items(), desc="Processing target areas", unit="area"):
        if not las_files:
            print(f"No valid LAS files for {target_fp}. Skipping.")
            continue

        clean_target_fp = os.path.splitext(target_fp)[0]
        final_output_file = os.path.join(run_merged_dir, f"{clean_target_fp}.las")

        if os.path.exists(final_output_file):
            print(f"Skipping {target_fp}: Already processed.")
            continue

        footprint_path = os.path.join(target_footprint_dir, target_fp if target_fp.endswith('.gpkg') else f"{target_fp}.gpkg")
        if not os.path.exists(footprint_path):
            print(f"Footprint file {footprint_path} not found. Skipping.")
            continue

        gdf = gpd.read_file(footprint_path)
        temp_dir = os.path.join(run_merged_dir, target_fp, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        processed_chunks = []
        process_args = []

        for input_file in las_files:
            if not is_utm_crs(input_file):
                # Handle both .las and .laz extensions
                base_name = os.path.basename(input_file)
                base_name = base_name.replace('.las', '_utm.las').replace('.laz', '_utm.las')
                utm_output_file = os.path.join(temp_dir, base_name)
                input_file = reproject_las(input_file, utm_output_file)

            ref_scale, ref_offset, ref_crs = get_las_header(input_file)

            if gdf.crs.to_epsg() != ref_crs:
                gdf = gdf.to_crs(epsg=ref_crs)

            target_geom_wkt = wkt_dumps(shape(gdf.geometry.iloc[0]))
            chunks = create_chunks_from_wkt(target_geom_wkt, chunk_size)

            if max_elev:

                all_z = laspy.read(input_file).z
                max_z = np.quantile(all_z, max_elev)
                min_z = np.quantile(all_z, 1 - max_elev)

            else:
                all_z = laspy.read(input_file).z
                max_z = np.max(all_z)
                min_z = np.min(all_z)


            for chunk in chunks:
                process_args.append((input_file, chunk, temp_dir, max_z, min_z, sor_knn, sor_multiplier, ref_scale, ref_offset, ref_crs))

        with tqdm(total=len(process_args), desc=f"Processing {target_fp}", unit="chunk") as pbar:
            with Pool(processes=num_workers) as pool:
                for processed_chunk in pool.imap_unordered(process_chunk_wrapper, process_args):
                    if processed_chunk:
                        processed_chunks.append(processed_chunk)
                    pbar.update(1)

        if processed_chunks:
            merge_and_crop_chunks(processed_chunks, target_geom_wkt, final_output_file)
            print(f"Final processed LAS file saved: {final_output_file}")
        else:
            print(f"No processed chunks available for {target_fp}.")

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        target_fp_dir = os.path.join(run_merged_dir, target_fp)
        if os.path.isdir(target_fp_dir) and not os.listdir(target_fp_dir):
            os.rmdir(target_fp_dir)

    print(f"\nProcessing completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.")


def preprocess_all(conf):
    global config
    config = conf

    print("\n========== Starting Preprocessing ==========")
    start = time.time()

    run_name = config.run_name

    os.makedirs(os.path.join(config.preprocessed_dir, run_name), exist_ok=True)
    os.makedirs(os.path.join(config.results_dir, run_name), exist_ok=True)

    gdfs = os.listdir(config.target_area_dir)
    for gdf in gdfs:
        gdf_path = os.path.join(config.target_area_dir, gdf)
        gdf_loaded = gpd.read_file(gdf_path)
        if len(gdf_loaded) > 1:
            print("\n--- Target areas are multi-geometry. Splitting into separate files ---")
            for gdf_name in os.listdir(config.target_area_dir):
                split_gpkg(os.path.join(config.target_area_dir, gdf_name), config.target_area_dir, field_name=config.target_name_field)
            break

    
    out_dir = os.path.join(config.preprocessed_dir, config.run_name)

    print("\n--- Matching footprints to LAS files ---")
    target_dict = match_footprints(
        target_footprint_dir=config.target_area_dir,
        las_footprint_dir=config.las_footprints_dir,
        las_file_dir=config.las_files_dir,
        out_dir=out_dir,
        threshold=config.overlap,
        filter_date=config.filter_date,
        start_date=config.start_date,
        end_date=config.end_date
    )

    print("\n--- Merging and Cleaning LAS files ---")
    merge_and_clean_las(
        target_footprint_dir=config.target_area_dir,
        las_dict=target_dict,
        preprocessed_dir=config.preprocessed_dir,
        max_elev=config.max_elevation_threshold,
        sor_knn=config.knn,
        sor_multiplier=config.multiplier,
        num_workers=config.num_workers,
        run_name=run_name,
        chunk_size=config.chunk_size
    )

    print(f"\nPreprocessing completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n")
