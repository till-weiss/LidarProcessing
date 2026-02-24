import os
import time
import json
import glob
import pdal
import laspy
import numpy as np
import subprocess
from datetime import timedelta
from tqdm import tqdm
from scipy.spatial import KDTree
import rasterio
import signal
import functools
import shutil 
from matplotlib import pyplot as plt
from rasterio.warp import reproject, Resampling
import laspy
from shapely.geometry import box
from multiprocessing import get_context
import multiprocessing
from shapely.wkt import loads as wkt_loads, dumps as wkt_dumps

from core.processing_windowed import create_chunks_from_wkt, process_chunk_to_dsm, process_chunk_to_dem_with_laz_outputs, merge_chunks, merge_laz_chunks, fill_nodata_raster_inplace
from core.icp_alignment import (
    read_xyz_laspy,
    run_icp,
    compute_overlap_bbox_from_headers,
    crop_bbox_with_pdal,
    apply_transformation_with_pdal,
    merge_two_laz_with_pdal,
    metrics_from_T,
    write_icp_report_txt,
    append_json_ledger,
)
from pathlib import Path



def check_resolution(las_file, resolution, method="sampling", num_samples=10000):
    """
    Checks if the DSM resolution is appropriate based on point cloud density.

    Parameters:
        las_file (str): Path to the LAS/LAZ file.
        resolution (float): Desired DSM resolution.
        method (str): "sampling" (nearest neighbor) or "density" (Poisson estimate).
        num_samples (int): Number of random samples for nearest neighbor method.

    Returns:
        float: Estimated average point spacing.
        bool: Whether the resolution is appropriate.
    """
    with laspy.open(las_file) as file:
        point_cloud = file.read()
        points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).T

    if len(points) == 0:
        raise ValueError(f"Point cloud {las_file} is empty.")

    if method == "sampling":
        num_samples = min(num_samples, len(points))
        sampled_points = points[np.random.choice(len(points), num_samples, replace=False)]
        tree = KDTree(points)
        distances, _ = tree.query(sampled_points, k=2)
        avg_distance = np.mean(distances[:, 1])  # Ignore self-distance

    elif method == "density":
        bbox_volume = np.prod(points.max(axis=0) - points.min(axis=0))
        density = len(points) / bbox_volume if bbox_volume > 0 else float('inf')
        avg_distance = (1 / density) ** (1 / 3)

    else:
        raise ValueError("Invalid method. Choose 'sampling' or 'density'.")

    #if avg_distance > resolution:
        #print(f"Warning: DSM resolution ({resolution}m) is finer than average point spacing ({avg_distance:.3f}m). "
              #f"This may cause interpolation gaps.")

    return avg_distance, avg_distance <= resolution

def get_las_footprint_wkt(las_file):
    """Extracts the WKT footprint (bounding box) from a LAS file."""
    
    with laspy.open(las_file) as las:
        header = las.header
        min_x, min_y, max_x, max_y = header.min[0], header.min[1], header.max[0], header.max[1]

    # Create a bounding box polygon
    footprint = box(min_x, min_y, max_x, max_y)
    return footprint.wkt  # Convert to WKT format


def process_dsm_chunk_wrapper(args):
    try:
        las_file, large_chunk, small_chunk, output_dir, resolution = args
        return process_chunk_to_dsm(las_file, large_chunk, small_chunk, output_dir, resolution)
    except Exception as e:
        return print(f"Error processing chunk for {las_file}: {e}")

def generate_dsm(input_folder, output_folder, run_name, method, resolution, chunk_size, chunk_overlap, num_workers, fill_gaps=True):

    final_output_folder = os.path.join(output_folder, run_name, 'DSM')
    os.makedirs(final_output_folder, exist_ok=True)
    temp_folder = os.path.join(final_output_folder, "temp")
    os.makedirs(temp_folder, exist_ok=True)

    start_time = time.time()
    las_files = glob.glob(os.path.join(input_folder, run_name, "*.las")) + \
                glob.glob(os.path.join(input_folder, run_name, "*.laz"))

    if not las_files:
        print("No LAS/LAZ files found. Exiting DSM generation.")
        return

    for las_file in tqdm(las_files, desc="Processing LAS files", unit="file"):
        target_wkt = get_las_footprint_wkt(las_file)
        #avg_spacing, is_resolution_ok = check_resolution(las_file, resolution, method)
        #if not is_resolution_ok:
        #    print(f"Warning: DSM resolution ({resolution}m) is finer than avg spacing ({avg_spacing:.3f}m).")


        chunk_tasks = []

        base_name = os.path.splitext(os.path.basename(las_file))[0]
        temp_dsm_dir = os.path.join(temp_folder, base_name)
        final_dsm_path = os.path.join(final_output_folder, f"{base_name}_DSM.tif")

        if not os.path.exists(final_dsm_path):

            print(f'saving temp files to {temp_dsm_dir}')

            base_name = os.path.splitext(os.path.basename(las_file))[0]
            temp_dsm_dir = os.path.join(temp_folder, base_name)
            os.makedirs(temp_dsm_dir, exist_ok=True)

            large_chunks, small_chunks = create_chunks_from_wkt(target_wkt, chunk_size, chunk_overlap)
            for large_chunk, small_chunk in zip(large_chunks, small_chunks):
                chunk_tasks.append((las_file, large_chunk, small_chunk, temp_dsm_dir, resolution))

            with multiprocessing.Pool(processes=num_workers) as pool:
                list(tqdm(
                    pool.imap_unordered(process_dsm_chunk_wrapper, chunk_tasks),
                    total=len(chunk_tasks),
                    desc="Processing DSM Chunks"))


            chunk_files = sorted(glob.glob(os.path.join(temp_dsm_dir, "*.tif")))
            if not chunk_files:
                print(f"No DSM chunks found for {base_name}. Skipping.")
                continue

            merged_dsm = merge_chunks(chunk_files, final_dsm_path)
            if fill_gaps and merged_dsm:
                filled_dsm_path = os.path.join(temp_dsm_dir, f"{base_name}_filled.tif")
                try:
                    shutil.copy2(merged_dsm, filled_dsm_path)
                    fill_nodata_raster_inplace(filled_dsm_path, max_distance=10, smoothing_iterations=2)
                    os.replace(filled_dsm_path, final_dsm_path)
                except Exception as exc:
                    print(f"[WARNING] DSM fill_nodata failed, using unfilled raster: {exc}")

            #read output file for plotting
            with rasterio.open(final_dsm_path) as src:
                dsm_data = src.read(1)
                dsm_nodata = src.nodata if src.nodata is not None else np.nan
                dsm_data = np.where(dsm_data == dsm_nodata, np.nan, dsm_data)

            # Plot the merged DSM and save as PNG
            plt.figure(figsize=(10, 10))
            plt.imshow(dsm_data, cmap='terrain', vmin=np.nanpercentile(dsm_data, 2), vmax=np.nanpercentile(dsm_data, 98))
            plt.colorbar(label='Elevation (m)')
            plt.title(f'DSM: {base_name}')
            plt.axis('off')
            plt.savefig(os.path.join(final_output_folder, f"{base_name}_DSM.png"), bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close()  # Ensure we close the plot to free memory


            shutil.rmtree(temp_dsm_dir, ignore_errors=True)

        else:
            print(f"Skipping {base_name}: File already exists.")

    elapsed_time = timedelta(seconds=int(time.time() - start_time))
    print(f"\nDSM generation completed in {elapsed_time}.")


def process_dtm_chunk_wrapper(args):
    (las_file, large_chunk, small_chunk, output_dir, cleaned_chunk_dir,
     classified_chunk_dir, threshold, scalar, slope, window, rigidness,
     iterations, resolution, time_step, cloth_resolution, fill_gaps,
     filter_smrf, filter_csf) = args

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


def generate_dtm(input_folder, output_folder, run_name, resolution, chunk_size, fill_gaps, num_workers, method, chunk_overlap, threshold, scalar, slope, window, rigidness, iterations, time_step, cloth_resolution, filter_smrf, filter_csf):

    final_output_folder = os.path.join(output_folder, run_name, 'DTM')
    os.makedirs(final_output_folder, exist_ok=True)
    temp_folder = os.path.join(final_output_folder, "temp")
    os.makedirs(temp_folder, exist_ok=True)

    start_time = time.time()
    las_files = glob.glob(os.path.join(input_folder, run_name, "*.las")) +                 glob.glob(os.path.join(input_folder, run_name, "*.laz"))

    if not las_files:
        print("No LAS/LAZ files found. Exiting DTM generation.")
        return

    for las_file in tqdm(las_files, desc="Processing LAS files", unit="file"):

        base_name = os.path.splitext(os.path.basename(las_file))[0]
        temp_dtm_dir = os.path.join(temp_folder, base_name)
        final_dtm_path = os.path.join(final_output_folder, f"{base_name}_DTM.tif")

        cleaned_out_dir = os.path.join(output_folder, run_name, "CLEANED_LAZ")
        cleaned_chunk_dir = os.path.join(cleaned_out_dir, "chunks", base_name)
        classified_out_dir = os.path.join(output_folder, run_name, "CLASSIFIED_LAZ")
        classified_chunk_dir = os.path.join(classified_out_dir, "chunks", base_name)
        os.makedirs(cleaned_chunk_dir, exist_ok=True)
        os.makedirs(classified_chunk_dir, exist_ok=True)

        if not os.path.exists(final_dtm_path):

            target_wkt = get_las_footprint_wkt(las_file)
            avg_spacing, is_resolution_ok = check_resolution(las_file, resolution, method)
            if not is_resolution_ok:
                print(f"Warning: DTM resolution ({resolution}m) is finer than avg spacing ({avg_spacing:.3f}m).")

            os.makedirs(temp_dtm_dir, exist_ok=True)

            large_chunks, small_chunks = create_chunks_from_wkt(
                target_wkt,
                chunk_size=chunk_size,
                overlap=chunk_overlap
            )

            chunk_tasks = []
            for large_chunk, small_chunk in zip(large_chunks, small_chunks):
                chunk_tasks.append((
                    las_file, large_chunk, small_chunk, temp_dtm_dir,
                    cleaned_chunk_dir, classified_chunk_dir,
                    threshold, scalar, slope, window, rigidness, iterations,
                    resolution, time_step, cloth_resolution,
                    fill_gaps, filter_smrf, filter_csf
                ))

            print(f"[DEBUG] {base_name}: DTM n_chunk_tasks={len(chunk_tasks)}")
            print(f"[DEBUG] {base_name}: cleaned chunks -> {cleaned_chunk_dir}")
            print(f"[DEBUG] {base_name}: classified chunks -> {classified_chunk_dir}")

            with multiprocessing.Pool(processes=num_workers) as pool:
                list(tqdm(
                    pool.imap_unordered(process_dtm_chunk_wrapper, chunk_tasks),
                    total=len(chunk_tasks),
                    desc="Processing DTM Chunks"))

            chunk_files = sorted(glob.glob(os.path.join(temp_dtm_dir, "*.tif")))
            if not chunk_files:
                print(f"No DTM chunks found for {base_name}. Skipping.")
                continue

            merged_dtm = merge_chunks(chunk_files, final_dtm_path)

            if fill_gaps and merged_dtm:
                filled_dtm_path = os.path.join(temp_dtm_dir, f"{base_name}_filled.tif")
                try:
                    shutil.copy2(merged_dtm, filled_dtm_path)
                    fill_nodata_raster_inplace(filled_dtm_path, max_distance=10, smoothing_iterations=2)
                    os.replace(filled_dtm_path, final_dtm_path)
                except Exception as exc:
                    print(f"[WARNING] DTM fill_nodata failed, using unfilled raster: {exc}")

            cleaned_chunk_files = sorted(glob.glob(os.path.join(cleaned_chunk_dir, "*_cleaned.laz")))
            classified_chunk_files = sorted(glob.glob(os.path.join(classified_chunk_dir, "*_classified.laz")))
            ground_chunk_files = sorted(glob.glob(os.path.join(classified_chunk_dir, "*_ground.laz")))

            print(f"[DEBUG] {base_name}: cleaned/classified/ground chunk files = "
                  f"{len(cleaned_chunk_files)}/{len(classified_chunk_files)}/{len(ground_chunk_files)}")

            ground_merged_path = os.path.join(classified_out_dir, f"{base_name}_ground_merged.laz")
            merge_laz_chunks(cleaned_chunk_files, os.path.join(cleaned_out_dir, f"{base_name}_cleaned_merged.laz"))
            merge_laz_chunks(classified_chunk_files, os.path.join(classified_out_dir, f"{base_name}_classified_merged.laz"))
            merge_laz_chunks(ground_chunk_files, ground_merged_path)

            if os.path.exists(ground_merged_path) and os.path.getsize(ground_merged_path) > 0:
                try:
                    with laspy.open(ground_merged_path) as ground_las:
                        ground_points = ground_las.read()
                        unique_classes = np.unique(ground_points.classification)
                    if len(unique_classes) > 0 and not np.all(unique_classes == 2):
                        print(f"[WARNING] Ground merged file has non-ground classes: {unique_classes}")
                except Exception as exc:
                    print(f"[WARNING] Could not validate ground classes for {ground_merged_path}: {exc}")

            with rasterio.open(final_dtm_path) as src:
                dtm_data = src.read(1)
                dtm_nodata = src.nodata if src.nodata is not None else np.nan
                dtm_data = np.where(dtm_data == dtm_nodata, np.nan, dtm_data)

            plt.figure(figsize=(10, 10))
            plt.imshow(dtm_data, cmap='terrain', vmin=np.nanpercentile(dtm_data, 2), vmax=np.nanpercentile(dtm_data, 98))
            plt.colorbar(label='Elevation (m)')
            plt.title(f'DTM: {base_name}')
            plt.axis('off')
            plt.savefig(os.path.join(final_output_folder, f"{base_name}_DTM.png"), bbox_inches='tight', pad_inches=0.1)
            plt.close()

            # keep temp rasters for debugging
            # shutil.rmtree(temp_dtm_dir, ignore_errors=True)

        else:
            print(f"Skipping {base_name}: DTM already exists.")

    elapsed_time = timedelta(seconds=int(time.time() - start_time))
    print(f"\nDTM generation completed in {elapsed_time}.")


def generate_chm(input_folder, output_folder, run_name):
    dsm_folder = os.path.join(input_folder, run_name, "DSM")
    dtm_folder = os.path.join(input_folder, run_name, "DTM")
    chm_folder = os.path.join(output_folder, run_name, "CHM")
    os.makedirs(chm_folder, exist_ok=True)

    dsm_files = glob.glob(os.path.join(dsm_folder, "*.tif"))

    print("\nStarting CHM generation")
    start_time = time.time()

    if not dsm_files:
        print(f"No DSM files found in {dsm_folder}. Exiting CHM generation.")
        return

    for dsm_path in tqdm(dsm_files, desc="Processing CHMs", unit="file"):
        try:
            base_name = os.path.splitext(os.path.basename(dsm_path))[0].replace("_DSM", "")
            dtm_path = os.path.join(dtm_folder, f"{base_name}_DTM.tif")
            chm_output_path = os.path.join(chm_folder, f"{base_name}_CHM.tif")

            #if os.path.exists(chm_output_path):
            #    print(f"Skipping {base_name}: CHM already exists.")
            #    continue

            if not os.path.exists(dtm_path):
                print(f"Skipping {base_name}: Corresponding DTM not found.")
                continue

            with rasterio.open(dsm_path) as dsm_src, rasterio.open(dtm_path) as dtm_src:
                # Read DSM
                dsm = dsm_src.read(1)
                dsm_mask = dsm_src.read_masks(1)
                dsm_meta = dsm_src.meta.copy()
                dsm_nodata = dsm_src.nodata if dsm_src.nodata is not None else np.nan

                # Prepare DTM to be aligned to DSM grid
                dtm_aligned = np.full((dsm_src.height, dsm_src.width), dsm_nodata, dtype=np.float32)

                reproject(
                    source=rasterio.band(dtm_src, 1),
                    destination=dtm_aligned,
                    src_transform=dtm_src.transform,
                    src_crs=dtm_src.crs,
                    dst_transform=dsm_src.transform,
                    dst_crs=dsm_src.crs,
                    resampling=Resampling.bilinear,
                    src_nodata=dtm_src.nodata,
                    dst_nodata=dsm_nodata
                )

                #Mask DTM where dsm_mask is 0
                dtm_aligned[dsm_mask == 0] = dsm_nodata

                # Compute CHM
                chm = dsm - dtm_aligned
                chm[(dsm == dsm_nodata) | (dtm_aligned == dsm_nodata)] = dsm_nodata
                

                # Update metadata
                dsm_meta.update({
                    "dtype": "float32",
                    "nodata": dsm_nodata,
                    "compress": "lzw"
                })

                with rasterio.open(chm_output_path, "w", **dsm_meta) as chm_dst:
                    chm_dst.write(chm.astype(np.float32), 1)

                # plot merged_dsm and save as png
                # set dsm_nodata to np.nan for plotting
                chm = np.where(chm == dsm_nodata, np.nan, chm)

                plt.figure(figsize=(10, 10))
                plt.imshow(chm, cmap='viridis', vmin=np.nanpercentile(chm, 2), vmax=np.nanpercentile(chm, 98))
                plt.colorbar(label='Elevation (m)')
                plt.title(f'CHM: {base_name}')
                plt.axis('off')
                plt.savefig(os.path.join(chm_folder, f"{base_name}_CHM.png"), bbox_inches='tight', pad_inches=0.1)
                plt.close()

        except Exception as e:
            print(f"[ERROR] Failed to process {base_name}: {e}")

    elapsed_time = timedelta(seconds=int(time.time() - start_time))
    print(f"\nCHM generation completed in {elapsed_time}.")




def align_and_merge_ground_strips_incremental(ground_strip_paths, out_root, run_name, config):
    out_root = Path(out_root)
    ground_strip_paths = sorted([Path(p) for p in ground_strip_paths], key=lambda p: p.name)
    if not ground_strip_paths:
        raise ValueError("No ground strip paths provided for ICP alignment.")

    aligned_dir = out_root / run_name / "ICP_GROUND_ALIGNED"
    ref_dir = out_root / run_name / "ICP_GROUND_REF"
    reports_dir = out_root / run_name / "ICP_REPORTS"
    tmp_dir = reports_dir / "tmp"
    for d in [aligned_dir, ref_dir, reports_dir, tmp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    ledger_path = reports_dir / "icp_transforms.json"

    accumulated_ref = ref_dir / "ref_1.laz"
    shutil.copy2(ground_strip_paths[0], accumulated_ref)

    aligned_strips = [accumulated_ref]
    ref_steps = [accumulated_ref]

    for idx, src in enumerate(ground_strip_paths[1:], start=2):
        step_id = idx
        t0 = time.time()
        overlap_bbox = compute_overlap_bbox_from_headers(accumulated_ref, src, config.icp_overlap_buffer_m)
        reg = None
        T = np.eye(4)
        accepted = False
        reason = None
        moving_for_merge = src

        if overlap_bbox is None:
            accepted = False
            reason = "no_overlap"
        else:
            try:
                ref_for_icp = accumulated_ref
                src_for_icp = src
                if config.icp_use_overlap_crop:
                    ref_for_icp = tmp_dir / f"tmp_ref_subset_step_{step_id}.laz"
                    src_for_icp = tmp_dir / f"tmp_src_subset_step_{step_id}.laz"
                    crop_bbox_with_pdal(accumulated_ref, ref_for_icp, overlap_bbox)
                    crop_bbox_with_pdal(src, src_for_icp, overlap_bbox)

                reg, T = run_icp(
                    source_xyz=read_xyz_laspy(src_for_icp),
                    target_xyz=read_xyz_laspy(ref_for_icp),
                    voxel_size=config.icp_voxel_size,
                    max_dist=config.icp_max_corr_dist,
                    max_iters=config.icp_max_iters,
                )
                metrics = metrics_from_T(T)
                checks = []
                if reg.fitness < config.icp_min_fitness:
                    checks.append(f"fitness<{config.icp_min_fitness}")
                if abs(metrics["dz"]) > config.icp_max_abs_dz:
                    checks.append(f"abs(dz)>{config.icp_max_abs_dz}")
                if metrics["rotation_deg"] > config.icp_max_rotation_deg:
                    checks.append(f"rotation>{config.icp_max_rotation_deg}")

                if checks:
                    accepted = False
                    reason = "threshold_fail:" + ",".join(checks)
                else:
                    accepted = True
                    aligned_src = aligned_dir / f"{src.stem}_icp.laz"
                    apply_transformation_with_pdal(src, aligned_src, T)
                    moving_for_merge = aligned_src
                    aligned_strips.append(aligned_src)
            except Exception as exc:
                accepted = False
                reason = f"icp_error:{exc}"

        if not accepted:
            if config.icp_fail_policy == "raise":
                raise RuntimeError(f"ICP step {step_id} failed: {reason}")
            if config.icp_fail_policy == "skip_strip":
                record = {
                    "step": step_id, "ref_in": str(accumulated_ref), "src_in": str(src),
                    "overlap_bbox": overlap_bbox, "accepted": False, "reason": reason,
                    "runtime_sec": time.time() - t0
                }
                append_json_ledger(ledger_path, record)
                write_icp_report_txt(
                    report_path=reports_dir / f"icp_step_{step_id}.txt",
                    step_id=step_id, ref_path=accumulated_ref, src_path=src,
                    params={"fail_policy": config.icp_fail_policy},
                    reg=reg, T=T, runtime_sec=time.time() - t0,
                    accepted=False, reason=reason, overlap_bbox=overlap_bbox
                )
                continue
            moving_for_merge = src

        out_ref = ref_dir / f"ref_1to{step_id}.laz"
        merge_two_laz_with_pdal(accumulated_ref, moving_for_merge, out_ref)
        accumulated_ref = out_ref
        ref_steps.append(out_ref)

        reg_fitness = getattr(reg, "fitness", None)
        reg_rmse = getattr(reg, "inlier_rmse", None)
        m = metrics_from_T(T)
        record = {
            "step": step_id,
            "ref_in": str(ref_steps[-2]),
            "src_in": str(src),
            "overlap_bbox": overlap_bbox,
            "accepted": accepted,
            "reason": reason,
            "fitness": reg_fitness,
            "inlier_rmse": reg_rmse,
            "dx": m["dx"], "dy": m["dy"], "dz": m["dz"], "rotation_deg": m["rotation_deg"],
            "T_flat": [float(v) for v in T.reshape(-1)],
            "runtime_sec": time.time() - t0,
            "used_moving_for_merge": str(moving_for_merge),
            "out_ref": str(out_ref),
        }
        append_json_ledger(ledger_path, record)
        write_icp_report_txt(
            report_path=reports_dir / f"icp_step_{step_id}.txt",
            step_id=step_id, ref_path=ref_steps[-2], src_path=src,
            params={
                "voxel_size": config.icp_voxel_size,
                "max_dist": config.icp_max_corr_dist,
                "max_iters": config.icp_max_iters,
                "overlap_buffer_m": config.icp_overlap_buffer_m,
                "min_fitness": config.icp_min_fitness,
                "max_abs_dz": config.icp_max_abs_dz,
                "max_rotation_deg": config.icp_max_rotation_deg,
                "fail_policy": config.icp_fail_policy,
            },
            reg=reg, T=T, runtime_sec=time.time() - t0,
            accepted=accepted, reason=reason, overlap_bbox=overlap_bbox
        )

    return {
        "aligned_strips": aligned_strips,
        "ref_steps": ref_steps,
        "final_ref": accumulated_ref,
        "reports_dir": reports_dir,
        "ledger_path": ledger_path,
    }

def process_all(config):
    """
    Runs DSM generation using cleaned LAS files.

    Reads from: `config.preprocessed_dir`
    Saves to: `config.results_dir / run_name / .../`
    """
    print('Starting Processing ...')

    start_time = time.time()

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
            chunk_overlap=config.chunk_overlap
        )
    
    

    if config.create_DEM:
        print("\n========== Starting DEM Generation ==========")
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
            rigidness = config.csf_rigidness,
            time_step=config.csf_time_step,
            cloth_resolution=config.csf_cloth_resolution,
            iterations = config.csf_iterations,
            num_workers= config.num_workers,
            chunk_overlap=config.chunk_overlap,
            filter_smrf=config.smrf_filter,
            filter_csf=config.csf_filter,
            threshold=config.threshold
        )


        if getattr(config, "enable_strip_icp", False):
            classified_dir = Path(config.results_dir) / config.run_name / "CLASSIFIED_LAZ"
            ground_strip_paths = sorted(classified_dir.glob("*_ground_merged.laz"), key=lambda p: p.name)
            if len(ground_strip_paths) < 2:
                print("[INFO] Strip ICP enabled but fewer than 2 ground merged strips were found; skipping ICP.")
            else:
                print(f"[INFO] Running strip ICP on {len(ground_strip_paths)} ground merged strips.")
                icp_result = align_and_merge_ground_strips_incremental(
                    ground_strip_paths=ground_strip_paths,
                    out_root=Path(config.results_dir),
                    run_name=config.run_name,
                    config=config
                )
                print(f"[INFO] ICP final reference: {icp_result['final_ref']}")
                print(f"[INFO] ICP reports: {icp_result['reports_dir']}")

    if config.create_CHM:
        print("\n========== Starting CHM Generation ==========")
        generate_chm(
            input_folder=config.results_dir,
            output_folder=config.results_dir,
            run_name=config.run_name
        )

    elapsed_time = timedelta(seconds=int(time.time() - start_time))
    print(f"\n DEM generation completed in {elapsed_time}.\n")