import os
import time
import json
import shutil
import socket
import sys
import uuid
import hashlib
from datetime import timedelta
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

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
from core.icp_alignment import (
    read_xyz_laspy,
    run_icp,
    compute_overlap_bbox_from_headers,
    crop_bbox_with_pdal,
    apply_transformation_with_pdal,
    merge_two_laz_with_pdal,
    append_jsonl_record,
    extract_las_crs_info,
    compute_crop_diagnostics,
    get_open3d_info,
)

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


def _align_and_merge_strip_files_incremental(strip_files, final_output_file, target_name, run_name, config):
    """Align cleaned strip LAS files incrementally before final merge."""
    strip_files = sorted([Path(p) for p in strip_files], key=lambda p: p.name)
    if len(strip_files) == 1:
        shutil.copy2(strip_files[0], final_output_file)
        return final_output_file

    report_dir = Path(config.results_dir) / run_name / "ICP_PREPROCESS_REPORT"
    aligned_dir = report_dir / "aligned_strips" / target_name
    ref_dir = report_dir / "refs" / target_name
    tmp_dir = report_dir / "tmp" / target_name
    for d in [aligned_dir, ref_dir, tmp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{uuid.uuid4().hex[:8]}"
    jsonl_path = report_dir / f"icp_attempts_{run_id}.jsonl"
    params = {
        "voxel_size": config.icp_voxel_size,
        "max_dist": config.icp_max_corr_dist,
        "max_iters": config.icp_max_iters,
        "overlap_buffer_m": config.icp_overlap_buffer_m,
        "min_fitness": config.icp_min_fitness,
        "max_abs_dz": config.icp_max_abs_dz,
        "max_rotation_deg": config.icp_max_rotation_deg,
        "strict_crs_check": getattr(config, "strict_crs_check", True),
        "min_points": getattr(config, "min_points", 500),
        "max_pre_dxy": getattr(config, "max_pre_dxy", 100.0),
        "max_pre_dz": getattr(config, "max_pre_dz", 5.0),
    }
    config_hash = hashlib.sha256(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()
    o3d_info = get_open3d_info()
    if getattr(config, "debug_mode", False) and not o3d_info["open3d_available"]:
        raise RuntimeError("Open3D is unavailable and debug_mode=True; aborting preprocess ICP.")

    accumulated_ref = ref_dir / "ref_1.las"
    shutil.copy2(strip_files[0], accumulated_ref)

    for idx, src in enumerate(strip_files[1:], start=2):
        t0 = time.time()
        attempt = {
            "run_id": run_id,
            "attempt_id": f"{idx}_{uuid.uuid4().hex}",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "sys.executable": sys.executable,
            "python_version": sys.version,
            "open3d_available": o3d_info["open3d_available"],
            "open3d_version": o3d_info["open3d_version"],
            "config_hash": config_hash,
            "target_name": target_name,
            "ref_path": str(accumulated_ref),
            "src_path": str(src),
            "accepted": False,
            "rejected_by": [],
        }

        overlap_bbox = compute_overlap_bbox_from_headers(accumulated_ref, src, config.icp_overlap_buffer_m)
        attempt["overlap_bbox"] = overlap_bbox
        ref_crs = extract_las_crs_info(accumulated_ref)
        src_crs = extract_las_crs_info(src)
        attempt["crs"] = {"ref": ref_crs, "src": src_crs}

        moving_for_merge = src
        T = np.eye(4)
        reg = None

        if overlap_bbox is None:
            attempt["rejected_by"].append("precheck_fail:no_overlap")

        if getattr(config, "strict_crs_check", True):
            ref_present = ref_crs.get("present")
            src_present = src_crs.get("present")
            if ref_present != src_present:
                attempt["rejected_by"].append("precheck_fail:missing_crs")
            if ref_present and src_present:
                ref_epsg = ref_crs.get("epsg") or ref_crs.get("horizontal")
                src_epsg = src_crs.get("epsg") or src_crs.get("horizontal")
                if ref_epsg != src_epsg:
                    attempt["rejected_by"].append("precheck_fail:crs_mismatch")
                if tuple(ref_crs.get("units") or []) != tuple(src_crs.get("units") or []):
                    attempt["rejected_by"].append("precheck_fail:unit_mismatch")

        if not attempt["rejected_by"] and overlap_bbox is not None:
            try:
                ref_subset = tmp_dir / f"ref_subset_{idx}.las"
                src_subset = tmp_dir / f"src_subset_{idx}.las"
                crop_bbox_with_pdal(accumulated_ref, ref_subset, overlap_bbox)
                crop_bbox_with_pdal(src, src_subset, overlap_bbox)
                xyz_ref = read_xyz_laspy(ref_subset)
                xyz_src = read_xyz_laspy(src_subset)
                diag = compute_crop_diagnostics(xyz_ref, xyz_src)
                attempt.update(diag)

                if diag["n_ref_crop"] < config.min_points or diag["n_src_crop"] < config.min_points:
                    attempt["rejected_by"].append("precheck_fail:empty_or_sparse_overlap")
                if diag.get("dxy_centroid_pre") is not None and diag["dxy_centroid_pre"] > config.max_pre_dxy:
                    attempt["rejected_by"].append("precheck_fail:max_pre_dxy")
                if diag.get("dz_median_pre") is not None and abs(diag["dz_median_pre"]) > config.max_pre_dz:
                    attempt["rejected_by"].append("precheck_fail:max_pre_dz")

                if not attempt["rejected_by"]:
                    reg, T = run_icp(xyz_src, xyz_ref, config.icp_voxel_size, config.icp_max_corr_dist, config.icp_max_iters)
                    attempt["fitness"] = float(reg.fitness)
                    attempt["inlier_rmse"] = float(reg.inlier_rmse)
                    if reg.fitness < config.icp_min_fitness:
                        attempt["rejected_by"].append("min_fitness")
                    dx, dy, dz = float(T[0, 3]), float(T[1, 3]), float(T[2, 3])
                    attempt.update({"dx": dx, "dy": dy, "dz": dz, "transform_matrix": [[float(v) for v in row] for row in T.tolist()]})
                    if abs(dz) > config.icp_max_abs_dz:
                        attempt["rejected_by"].append("max_abs_dz")

                    if not attempt["rejected_by"]:
                        aligned_src = aligned_dir / f"{src.stem}_icp.las"
                        apply_transformation_with_pdal(src, aligned_src, T)
                        moving_for_merge = aligned_src
                        attempt["accepted"] = True
            except Exception as exc:
                attempt["rejected_by"].append(f"icp_error:{exc}")

        out_ref = ref_dir / f"ref_1to{idx}.las"
        merge_two_laz_with_pdal(accumulated_ref, moving_for_merge, out_ref)
        accumulated_ref = out_ref
        attempt["used_moving_for_merge"] = str(moving_for_merge)
        attempt["out_ref"] = str(out_ref)
        attempt["runtime_sec"] = time.time() - t0
        append_jsonl_record(jsonl_path, attempt)

    shutil.copy2(accumulated_ref, final_output_file)
    return final_output_file

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

        strip_cleaned_files = []

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


            process_args = [
                (input_file, chunk, temp_dir, max_z, min_z, sor_knn, sor_multiplier, ref_scale, ref_offset, ref_crs)
                for chunk in chunks
            ]
            processed_chunks = []
            with tqdm(total=len(process_args), desc=f"Processing {os.path.basename(input_file)}", unit="chunk") as pbar:
                with Pool(processes=num_workers) as pool:
                    for processed_chunk in pool.imap_unordered(process_chunk_wrapper, process_args):
                        if processed_chunk:
                            processed_chunks.append(processed_chunk)
                        pbar.update(1)

            if processed_chunks:
                strip_out = os.path.join(temp_dir, f"{Path(input_file).stem}_cleaned_strip.las")
                merge_and_crop_chunks(processed_chunks, target_geom_wkt, strip_out)
                strip_cleaned_files.append(strip_out)

        if strip_cleaned_files:
            if getattr(config, "enable_strip_icp", False) and len(strip_cleaned_files) > 1:
                _align_and_merge_strip_files_incremental(
                    strip_files=strip_cleaned_files,
                    final_output_file=final_output_file,
                    target_name=clean_target_fp,
                    run_name=run_name,
                    config=config,
                )
            else:
                merge_and_crop_chunks(strip_cleaned_files, target_geom_wkt, final_output_file)
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
