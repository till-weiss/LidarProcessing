import os
import time
import glob
import shutil
import subprocess
import traceback
import multiprocessing
from datetime import timedelta

import numpy as np
import laspy
import rasterio
from tqdm import tqdm
from matplotlib import pyplot as plt
from shapely.geometry import box

from core.processing_windowed import (
    create_chunks_from_wkt,
    process_chunk_to_dsm,
    process_chunk_to_dem_with_laz_outputs,
    merge_chunks,
    merge_laz_chunks, fill_nodata_raster_inplace
)


# ----------------------------
# Utils
# ----------------------------

def get_las_footprint_wkt(las_file: str) -> str:
    """Extract footprint (bbox) from LAS header and return as WKT."""
    with laspy.open(las_file) as las:
        header = las.header
        min_x, min_y = header.min[0], header.min[1]
        max_x, max_y = header.max[0], header.max[1]
    return box(min_x, min_y, max_x, max_y).wkt


def _list_files_recursive(folder: str, max_show: int = 50) -> None:
    files = sorted(glob.glob(os.path.join(folder, "**", "*.*"), recursive=True))
    print(f"[DEBUG] Files under {folder}: {len(files)}")
    for f in files[:max_show]:
        print("   -", os.path.relpath(f, folder))


# ----------------------------
# DSM (DEBUG)
# ----------------------------

def process_dsm_chunk_wrapper(args):
    """
    DEBUG wrapper:
      - never swallows exceptions
      - prints return value
    """
    las_file, large_chunk, small_chunk, output_dir, resolution = args

    try:
        out = process_chunk_to_dsm(las_file, large_chunk, small_chunk, output_dir, resolution)
        print(f"[DEBUG] DSM chunk returned: {out}")
        return out
    except Exception as e:
        print(f"\n[DSM ERROR] Chunk failed for {os.path.basename(las_file)}")
        print("Reason:", repr(e))
        traceback.print_exc()
        raise


def generate_dsm(
    input_folder: str,
    output_folder: str,
    run_name: str,
    method: str,
    resolution: float,
    chunk_size: float,
    chunk_overlap: float,
    num_workers: int,
    fill_gaps: bool = True,
    debug_serial_first_chunk: bool = True,
    debug_disable_multiprocessing: bool = False,
):

    final_output_folder = os.path.join(output_folder, run_name, "DSM")
    os.makedirs(final_output_folder, exist_ok=True)

    temp_folder = os.path.join(final_output_folder, "temp")
    os.makedirs(temp_folder, exist_ok=True)

    start_time = time.time()

    las_files = (
        glob.glob(os.path.join(input_folder, run_name, "*.las")) +
        glob.glob(os.path.join(input_folder, run_name, "*.laz"))
    )

    print("[DEBUG] DSM input folder:", os.path.join(input_folder, run_name))
    print("[DEBUG] DSM found LAS/LAZ:", las_files)

    if not las_files:
        print("No LAS/LAZ files found. Exiting DSM generation.")
        return

    for las_file in tqdm(las_files, desc="Processing LAS files", unit="file"):

        base_name = os.path.splitext(os.path.basename(las_file))[0]
        temp_dsm_dir = os.path.join(temp_folder, base_name)
        os.makedirs(temp_dsm_dir, exist_ok=True)

        final_dsm_path = os.path.join(final_output_folder, f"{base_name}_DSM.tif")

        if os.path.exists(final_dsm_path):
            print(f"Skipping {base_name}: DSM already exists -> {final_dsm_path}")
            continue

        print(f"\n[DEBUG] saving temp files to {temp_dsm_dir}")

        target_wkt = get_las_footprint_wkt(las_file)
        large_chunks, small_chunks = create_chunks_from_wkt(target_wkt, chunk_size, chunk_overlap)

        chunk_tasks = [
            (las_file, large_chunk, small_chunk, temp_dsm_dir, resolution)
            for large_chunk, small_chunk in zip(large_chunks, small_chunks)
        ]

        print(f"[DEBUG] {base_name}: n_chunk_tasks = {len(chunk_tasks)}")

        if not chunk_tasks:
            print(f"[DEBUG] No chunk tasks created for {base_name}. Check create_chunks_from_wkt().")
            _list_files_recursive(temp_dsm_dir)
            continue

        # --- run first chunk serially (best debugging signal) ---
        if debug_serial_first_chunk:
            print("[DEBUG] Running first DSM chunk serially:")
            print("        task:", chunk_tasks[0])
            out0 = process_dsm_chunk_wrapper(chunk_tasks[0])
            print("[DEBUG] First chunk output:", out0)
            _list_files_recursive(temp_dsm_dir)

        # --- run the rest ---
        remaining_tasks = chunk_tasks[1:] if debug_serial_first_chunk else chunk_tasks

        if remaining_tasks:
            if debug_disable_multiprocessing or num_workers <= 1:
                print("[DEBUG] Running DSM chunks serially (no multiprocessing).")
                for t in tqdm(remaining_tasks, desc="Processing DSM Chunks (serial)"):
                    process_dsm_chunk_wrapper(t)
            else:
                print(f"[DEBUG] Running DSM chunks with multiprocessing (workers={num_workers}).")
                with multiprocessing.Pool(processes=num_workers) as pool:
                    list(tqdm(
                        pool.imap_unordered(process_dsm_chunk_wrapper, remaining_tasks),
                        total=len(remaining_tasks),
                        desc="Processing DSM Chunks"
                    ))

        # --- find chunk rasters (recursive!) ---
        chunk_files = sorted(glob.glob(os.path.join(temp_dsm_dir, "**", "*.tif"), recursive=True))
        print(f"[DEBUG] Found DSM chunk tif files: {len(chunk_files)}")

        if not chunk_files:
            print(f"No DSM chunks found for {base_name}. Skipping.")
            _list_files_recursive(temp_dsm_dir)
            continue

        # --- merge ---
        merged_dsm = merge_chunks(chunk_files, final_dsm_path)
        print("[DEBUG] merge_chunks returned:", merged_dsm)
        print("[DEBUG] final DSM should be:", final_dsm_path)

        if fill_gaps and merged_dsm:
            filled_dsm_path = os.path.join(temp_dsm_dir, f"{base_name}_filled.tif")
            try:
                shutil.copy2(merged_dsm, filled_dsm_path)
                fill_nodata_raster_inplace(filled_dsm_path, max_distance=10, smoothing_iterations=2)
                os.replace(filled_dsm_path, final_dsm_path)
            except Exception as exc:
                print(f"[WARNING] DSM fill_nodata failed, using unfilled raster: {exc}")

        # --- quick plot ---
        with rasterio.open(final_dsm_path) as src:
            dsm_data = src.read(1)
            nodata = src.nodata if src.nodata is not None else np.nan
            dsm_data = np.where(dsm_data == nodata, np.nan, dsm_data)

        plt.figure(figsize=(10, 10))
        plt.imshow(dsm_data, cmap="terrain",
                   vmin=np.nanpercentile(dsm_data, 2),
                   vmax=np.nanpercentile(dsm_data, 98))
        plt.colorbar(label="Elevation (m)")
        plt.title(f"DSM: {base_name}")
        plt.axis("off")
        plt.savefig(os.path.join(final_output_folder, f"{base_name}_DSM.png"),
                    bbox_inches="tight", pad_inches=0.1, dpi=300)
        plt.close()

        # keep temp for debugging? comment out if you want to inspect outputs
        # shutil.rmtree(temp_dsm_dir, ignore_errors=True)

    print(f"\nDSM generation completed in {timedelta(seconds=int(time.time() - start_time))}.")


# ----------------------------
# DTM (your code calls this under create_DEM)
# ----------------------------

def process_dtm_chunk_wrapper(args):
    """
    DEBUG-friendly: don't swallow exceptions.
    """
    (las_file, large_chunk, small_chunk, output_dir, cleaned_chunk_dir,
     classified_chunk_dir, threshold, scalar, slope, window, rigidness,
     iterations, resolution, time_step, cloth_resolution, fill_gaps,
     filter_smrf, filter_csf) = args

    try:
        return process_chunk_to_dem_with_laz_outputs(
            input_file=las_file,
            large_chunk_bbox=large_chunk,
            small_chunk_bbox=small_chunk,
            temp_dir=output_dir,
            cleaned_chunk_dir=cleaned_chunk_dir,
            classified_chunk_dir=classified_chunk_dir,
            scalar=scalar,
            threshold=threshold,
            slope=slope,
            window=window,
            rigidness=rigidness,
            iterations=iterations,
            resolution=resolution,
            time_step=time_step,
            cloth_resolution=cloth_resolution,
            fill_gaps=fill_gaps,
            filter_smrf=filter_smrf,
            filter_csf=filter_csf
        )
    except Exception as e:
        print(f"\n[DTM ERROR] Chunk failed for {os.path.basename(las_file)}")
        print("Reason:", repr(e))
        traceback.print_exc()
        raise


def generate_dtm(
    input_folder,
    output_folder,
    run_name,
    resolution,
    chunk_size,
    fill_gaps,
    num_workers,
    method,
    chunk_overlap,
    threshold,
    scalar,
    slope,
    window,
    rigidness,
    iterations,
    time_step,
    cloth_resolution,
    filter_smrf,
    filter_csf
):
    # unchanged logic except: recursive chunk search + debug prints
    final_output_folder = os.path.join(output_folder, run_name, "DTM")
    os.makedirs(final_output_folder, exist_ok=True)
    temp_folder = os.path.join(final_output_folder, "temp")
    os.makedirs(temp_folder, exist_ok=True)

    start_time = time.time()
    las_files = (
        glob.glob(os.path.join(input_folder, run_name, "*.las")) +
        glob.glob(os.path.join(input_folder, run_name, "*.laz"))
    )

    print("[DEBUG] DTM found LAS/LAZ:", las_files)

    if not las_files:
        print("No LAS/LAZ files found. Exiting DTM generation.")
        return

    for las_file in tqdm(las_files, desc="Processing LAS files", unit="file"):
        base_name = os.path.splitext(os.path.basename(las_file))[0]
        temp_dtm_dir = os.path.join(temp_folder, base_name)
        os.makedirs(temp_dtm_dir, exist_ok=True)

        final_dtm_path = os.path.join(final_output_folder, f"{base_name}_DTM.tif")

        cleaned_out_dir = os.path.join(output_folder, run_name, "CLEANED_LAZ")
        cleaned_chunk_dir = os.path.join(cleaned_out_dir, "chunks", base_name)
        classified_out_dir = os.path.join(output_folder, run_name, "CLASSIFIED_LAZ")
        classified_chunk_dir = os.path.join(classified_out_dir, "chunks", base_name)
        os.makedirs(cleaned_chunk_dir, exist_ok=True)
        os.makedirs(classified_chunk_dir, exist_ok=True)

        cleaned_merged_path = os.path.join(cleaned_out_dir, f"{base_name}_cleaned_merged.laz")
        classified_merged_path = os.path.join(classified_out_dir, f"{base_name}_classified_merged.laz")
        ground_merged_path = os.path.join(classified_out_dir, f"{base_name}_ground_merged.laz")

        dtm_exists = os.path.exists(final_dtm_path)
        laz_merged_exist = (
            os.path.exists(cleaned_merged_path) and
            os.path.exists(classified_merged_path) and
            os.path.exists(ground_merged_path)
        )

        if dtm_exists and laz_merged_exist:
            print(f"Skipping {base_name}: DTM and merged LAZ already exist.")
            continue

        if dtm_exists and not laz_merged_exist:
            print(f"[INFO] {base_name}: DTM exists but merged LAZ missing; recomputing chunks for LAZ export.")

        target_wkt = get_las_footprint_wkt(las_file)
        large_chunks, small_chunks = create_chunks_from_wkt(target_wkt, chunk_size=chunk_size, overlap=chunk_overlap)

        chunk_tasks = []
        for large_chunk, small_chunk in zip(large_chunks, small_chunks):
            chunk_tasks.append((
                las_file, large_chunk, small_chunk, temp_dtm_dir,
                cleaned_chunk_dir, classified_chunk_dir,
                threshold, scalar, slope, window, rigidness, iterations,
                resolution, time_step, cloth_resolution,
                fill_gaps, filter_smrf, filter_csf
            ))

        print(f"[DEBUG] {base_name}: DTM n_chunk_tasks = {len(chunk_tasks)}")

        with multiprocessing.Pool(processes=num_workers) as pool:
            list(tqdm(
                pool.imap_unordered(process_dtm_chunk_wrapper, chunk_tasks),
                total=len(chunk_tasks),
                desc="Processing DTM Chunks"
            ))

        chunk_files = sorted(glob.glob(os.path.join(temp_dtm_dir, "**", "*.tif"), recursive=True))
        print(f"[DEBUG] Found DTM chunk tif files: {len(chunk_files)}")

        if chunk_files:
            merged_dtm = merge_chunks(chunk_files, final_dtm_path)
            print("[DEBUG] merge_chunks returned:", merged_dtm)

            if fill_gaps and merged_dtm:
                filled_dtm_path = os.path.join(temp_dtm_dir, f"{base_name}_filled.tif")
                try:
                    shutil.copy2(merged_dtm, filled_dtm_path)
                    fill_nodata_raster_inplace(filled_dtm_path, max_distance=10, smoothing_iterations=2)
                    os.replace(filled_dtm_path, final_dtm_path)
                except Exception as exc:
                    print(f"[WARNING] DTM fill_nodata failed, using unfilled raster: {exc}")
        elif not os.path.exists(final_dtm_path):
            print(f"No DTM chunks found for {base_name}. Skipping.")
            _list_files_recursive(temp_dtm_dir)
            continue

        cleaned_chunk_files = sorted(glob.glob(os.path.join(cleaned_chunk_dir, "*_cleaned.laz")))
        classified_chunk_files = sorted(glob.glob(os.path.join(classified_chunk_dir, "*_classified.laz")))
        ground_chunk_files = sorted(glob.glob(os.path.join(classified_chunk_dir, "*_ground.laz")))

        print(
            f"[DEBUG] {base_name}: cleaned/classified/ground chunk files = "
            f"{len(cleaned_chunk_files)}/{len(classified_chunk_files)}/{len(ground_chunk_files)}"
        )

        merge_laz_chunks(cleaned_chunk_files, cleaned_merged_path)
        merge_laz_chunks(classified_chunk_files, classified_merged_path)
        merge_laz_chunks(ground_chunk_files, ground_merged_path)

        if os.path.exists(ground_merged_path) and os.path.getsize(ground_merged_path) > 0:
            try:
                with laspy.open(ground_merged_path) as ground_las:
                    unique_classes = np.unique(ground_las.read().classification)
                if len(unique_classes) > 0 and not np.all(unique_classes == 2):
                    print(f"[WARNING] Ground merged file has non-ground classes: {unique_classes}")
            except Exception as exc:
                print(f"[WARNING] Could not validate ground classes for {ground_merged_path}: {exc}")

        if not os.path.exists(final_dtm_path):
            print(f"[WARNING] Missing DTM raster after chunk processing: {final_dtm_path}")
            continue

        # plot
        with rasterio.open(final_dtm_path) as src:
            dtm_data = src.read(1)
            nodata = src.nodata if src.nodata is not None else np.nan
            dtm_data = np.where(dtm_data == nodata, np.nan, dtm_data)

        plt.figure(figsize=(10, 10))
        plt.imshow(dtm_data, cmap="terrain",
                   vmin=np.nanpercentile(dtm_data, 2),
                   vmax=np.nanpercentile(dtm_data, 98))
        plt.colorbar(label="Elevation (m)")
        plt.title(f"DTM: {base_name}")
        plt.axis("off")
        plt.savefig(os.path.join(final_output_folder, f"{base_name}_DTM.png"),
                    bbox_inches="tight", pad_inches=0.1)
        plt.close()

        # keep temp for debugging? comment out if you want to inspect outputs
        # shutil.rmtree(temp_dtm_dir, ignore_errors=True)

    print(f"\nDTM generation completed in {timedelta(seconds=int(time.time() - start_time))}.")


# ----------------------------
# Orchestrator
# ----------------------------

def process_all(config):
    print("Starting Processing ...")
    start_time = time.time()

    print("[DEBUG] flags:",
          "DSM", config.create_DSM,
          "DEM(flag actually runs DTM)", config.create_DEM,
          "CHM", config.create_CHM)

    if config.create_DSM:
        print("\n========== Starting DSM Generation ==========")
        generate_dsm(
            input_folder=config.preprocessed_dir,
            output_folder=config.results_dir,
            run_name=config.run_name,
            resolution=config.resolution,
            chunk_size=config.chunk_size,
            fill_gaps=config.fill_gaps,
            num_workers=config.num_workers,
            method=config.point_density_method,
            chunk_overlap=config.chunk_overlap,
            # DEBUG controls:
            debug_serial_first_chunk=True,
            debug_disable_multiprocessing=False
        )

    # NOTE: your code calls generate_dtm under create_DEM
    if config.create_DEM:
        print("\n========== Starting DTM Generation (called via create_DEM) ==========")
        generate_dtm(
            input_folder=config.preprocessed_dir,
            output_folder=config.results_dir,
            run_name=config.run_name,
            resolution=config.resolution,
            chunk_size=config.chunk_size,
            fill_gaps=config.fill_gaps,
            method=config.point_density_method,
            scalar=config.smrf_scalar,
            slope=config.smrf_slope,
            window=config.smrf_window_size,
            rigidness=config.csf_rigidness,
            time_step=config.csf_time_step,
            cloth_resolution=config.csf_cloth_resolution,
            iterations=config.csf_iterations,
            num_workers=config.num_workers,
            chunk_overlap=config.chunk_overlap,
            filter_smrf=config.smrf_filter,
            filter_csf=config.csf_filter,
            threshold=config.threshold
        )

    if config.create_CHM:
        print("\n========== Starting CHM Generation ==========")
        # keep your existing generate_chm here if needed
        # generate_chm(input_folder=config.results_dir, output_folder=config.results_dir, run_name=config.run_name)
        pass

    print(f"\nProcessing completed in {timedelta(seconds=int(time.time() - start_time))}")
