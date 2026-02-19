import laspy
import pdal
import numpy as np
import pandas as pd
import os
import rasterio
from rasterio.merge import merge
from osgeo import gdal
import shutil
import json
import subprocess
from shapely.geometry import box, shape
from shapely.wkt import loads as wkt_loads, dumps as wkt_dumps

def create_chunks_from_wkt(input_wkt, chunk_size=1000, overlap=0.2):
    """
    create processing chunks based on wkt geometry of target area, with overlap enlarged by x percent, 
    returns geometries for enlarged chunks and original chunk size, use original chunk size for cliping of results and large for cloth method
    """

    geom= wkt_loads(input_wkt)
    min_x, min_y, max_x, max_y = geom.bounds
    large_chunks = []
    orig_chunk_size = []

    enlarged_chunk_size = chunk_size * (1 + overlap)
    half_extra = (enlarged_chunk_size - chunk_size) / 2

    for x in np.arange(min_x, max_x, chunk_size):
        for y in np.arange(min_y, max_y, chunk_size):
            large_chunk_bbox = box(x - half_extra, y - half_extra, x + enlarged_chunk_size - half_extra, y + enlarged_chunk_size - half_extra)
            orig_chunk_box = box(x, y, x + chunk_size, y + chunk_size)
            if geom.intersects(orig_chunk_box):
                large_chunks.append(large_chunk_bbox)
                orig_chunk_size.append(orig_chunk_box)

    #check if chunk intersects with original geometry, just take intersecting chunks
    large_chunks = [chunk for chunk in large_chunks if geom.intersects(chunk)]
    orig_chunk_size = [chunk for chunk in orig_chunk_size if geom.intersects(chunk)]
    
    return large_chunks, orig_chunk_size

def process_chunk_to_dsm(input_file, large_chunk_bbox, small_chunk_bbox, temp_dir, resolution):

    chunk_file = os.path.join(
        temp_dir,
        f"{os.path.basename(input_file).replace('.las', '')}_chunk_{int(small_chunk_bbox.bounds[0])}_{int(small_chunk_bbox.bounds[1])}.tif"
    )

    pipeline = [
        {"type": "readers.las", "filename": input_file},
        {"type": "filters.crop", "polygon": wkt_dumps(large_chunk_bbox)},
        {"type": "filters.ferry", "dimensions": "Z=>Elevation"},
       # FOr testing  {
       #     "type": "filters.range",
       #     "limits": "Classification[0:0]"  # Use all points for initial DSM
       # },
        {"type": "filters.crop", "polygon": wkt_dumps(small_chunk_bbox)},
        {
            "type": "writers.gdal",
            "filename": chunk_file,
            "resolution": resolution,
            "output_type": "max",
            "nodata": -9999,
            "gdalopts": "COMPRESS=LZW"
        }
    ]

    try:
        #print("[INFO] Running PDAL pipeline...")
        pdal.pipeline.Pipeline(json.dumps(pipeline)).execute()
        #print("[INFO] PDAL execution completed.")
    except RuntimeError as e:
        print(f"[ERROR] PDAL execution failed: {e}. Empty chunk.")
        return None


    try:
        subprocess.run([
            "gdal_fillnodata.py",
            "-md", "10",
            "-si", "2",
            chunk_file,
            chunk_file
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        #print("[INFO] Nodata gaps filled with GDAL.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] GDAL fillnodata failed: {e}")

    return chunk_file

def process_chunk_to_dem(input_file, large_chunk_bbox, small_chunk_bbox, temp_dir, rigidness, iterations, resolution, time_step, cloth_resolution=1, fill_gaps=True, filter_smrf=False, scalar=None, slope=None, window=None, threshold=None, filter_csf=False):

    return process_chunk_to_dem_with_laz_outputs(
        input_file=input_file,
        large_chunk_bbox=large_chunk_bbox,
        small_chunk_bbox=small_chunk_bbox,
        temp_dir=temp_dir,
        cleaned_chunk_dir=temp_dir,
        classified_chunk_dir=temp_dir,
        rigidness=rigidness,
        iterations=iterations,
        resolution=resolution,
        time_step=time_step,
        cloth_resolution=cloth_resolution,
        fill_gaps=fill_gaps,
        filter_smrf=filter_smrf,
        scalar=scalar,
        slope=slope,
        window=window,
        threshold=threshold,
        filter_csf=filter_csf,
        save_cleaned_laz=False
    )


def _verify_point_cloud_output(path, label):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return True
    print(f"[WARNING] {label} was not written or is empty: {path}")
    return False


def process_chunk_to_dem_with_laz_outputs(input_file, large_chunk_bbox, small_chunk_bbox, temp_dir, cleaned_chunk_dir, classified_chunk_dir, rigidness, iterations, resolution, time_step, cloth_resolution=1, fill_gaps=True, filter_smrf=False, scalar=None, slope=None, window=None, threshold=None, filter_csf=False, save_cleaned_laz=True):

    chunk_file = os.path.join(
        temp_dir,
        f"{os.path.basename(input_file).replace('.las', '')}_chunk_{int(small_chunk_bbox.bounds[0])}_{int(small_chunk_bbox.bounds[1])}.tif"
    )

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(cleaned_chunk_dir, exist_ok=True)
    os.makedirs(classified_chunk_dir, exist_ok=True)

    base_name = os.path.basename(input_file).replace('.las', '').replace('.laz', '')
    chunk_id = f"{int(small_chunk_bbox.bounds[0])}_{int(small_chunk_bbox.bounds[1])}"
    cleaned_out_laz = os.path.join(cleaned_chunk_dir, f"{base_name}_chunk_{chunk_id}_cleaned.laz")
    classified_out_laz = os.path.join(classified_chunk_dir, f"{base_name}_chunk_{chunk_id}_classified.laz")
    ground_out_laz = os.path.join(classified_chunk_dir, f"{base_name}_chunk_{chunk_id}_ground.laz")

    print(f"[DEBUG] Chunk {chunk_id} outputs: cleaned={cleaned_out_laz}, classified={classified_out_laz}, ground={ground_out_laz}")

    cleaning_pipeline = [
        {"type": "readers.las", "filename": input_file},
        {"type": "filters.crop", "polygon": wkt_dumps(large_chunk_bbox)},
        {"type": "filters.crop", "polygon": wkt_dumps(small_chunk_bbox)},
        {"type": "writers.las", "filename": cleaned_out_laz, "compression": "laszip"}
    ]

    try:
        pdal.pipeline.Pipeline(json.dumps(cleaning_pipeline)).execute()
    except RuntimeError as e:
        print(f"[INFO] PDAL cleaned chunk pipeline failed: {e}. No points in chunk.")
        return None

    if save_cleaned_laz:
        _verify_point_cloud_output(cleaned_out_laz, "Cleaned chunk LAZ")

    classification_pipeline = [
        {"type": "readers.las", "filename": cleaned_out_laz},
    ]

    if filter_smrf:

        classification_pipeline.append({"type": "filters.smrf",
         "scalar": float(scalar),
         "slope": float(slope),
         "window": float(window)})
        
    if filter_csf:
        classification_pipeline.append(
        {"type": "filters.csf",
         "resolution": float(cloth_resolution),
         "rigidness": int(rigidness),
         "iterations": int(iterations),
         "step": float(time_step)})

    classification_pipeline += [
        {"type": "writers.las", "filename": classified_out_laz, "compression": "laszip"}
    ]

    try:
        pdal.pipeline.Pipeline(json.dumps(classification_pipeline)).execute()
    except RuntimeError as e:
        print(f"[INFO] PDAL classification pipeline failed: {e}. No points in chunk.")
        return None

    _verify_point_cloud_output(classified_out_laz, "Classified chunk LAZ")

    ground_pipeline = [
        {"type": "readers.las", "filename": classified_out_laz},
        {"type": "filters.range", "limits": "Classification[2:2]"},
        {"type": "writers.las", "filename": ground_out_laz, "compression": "laszip"}
    ]

    try:
        pdal.pipeline.Pipeline(json.dumps(ground_pipeline)).execute()
    except RuntimeError as e:
        print(f"[INFO] PDAL ground-only pipeline failed: {e}. No ground points in chunk.")
        return None

    if not _verify_point_cloud_output(ground_out_laz, "Ground-only chunk LAZ"):
        return None

    pipeline = [
        {"type": "readers.las", "filename": ground_out_laz},
        {"type": "filters.ferry", "dimensions": "Z=>Elevation"},
        {"type": "writers.gdal",
         "filename": chunk_file,
         "resolution": float(cloth_resolution),
         "output_type": "idw",
         "nodata": -9999,
         "gdalopts": "COMPRESS=LZW"}
    ]

    try:
        pdal.pipeline.Pipeline(json.dumps(pipeline)).execute()
    except RuntimeError as e:
        f"[INFO] PDAL execution failed: {e}. No Points in chunk after filterering."
        return None

    resampled_file = chunk_file.replace('.tif', '_resampled.tif')
    minx, miny, maxx, maxy = small_chunk_bbox.bounds

    try:
        subprocess.run([
            "gdalwarp",
            "-tr", str(resolution), str(resolution),
            "-r", "bilinear",
            "-tap",
            "-te", str(minx), str(miny), str(maxx), str(maxy),
            "-overwrite",
            chunk_file,
            resampled_file
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        shutil.move(resampled_file, chunk_file)

    except subprocess.CalledProcessError as e:
        print("[ERROR] gdalwarp failed:")
        print(e.stderr.decode('utf-8'))
        return None

    if fill_gaps:
        try:
            subprocess.run([
                "gdal_fillnodata.py",
                "-md", "100",
                "-si", "2",
                chunk_file,
                chunk_file
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] GDAL fillnodata failed: {e}")
            return None

    return chunk_file


def merge_laz_chunks(input_files, output_file):
    if not input_files:
        print(f"[WARNING] No LAZ chunk files found to merge for {output_file}")
        return None

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pipeline = [{"type": "readers.las", "filename": in_file} for in_file in input_files]
    pipeline.extend([
        {"type": "filters.merge"},
        {"type": "writers.las", "filename": output_file, "compression": "laszip"}
    ])

    try:
        pdal.pipeline.Pipeline(json.dumps(pipeline)).execute()
    except RuntimeError as e:
        print(f"[ERROR] Failed to merge LAZ chunks into {output_file}: {e}")
        return None

    if _verify_point_cloud_output(output_file, "Merged LAZ"):
        return output_file
    return None

def merge_chunks(input_files, output_file):
    """
    Merges multiple raster files into a single raster and saves the output.
    
    Parameters:
        input_files (list): List of file paths to raster files.
        output_file (str): Path to the output merged raster file.
    """
    
    # Open all raster files
    src_files = [rasterio.open(f) for f in input_files]
    
    # Merge rasters
    mosaic, out_transform = merge(src_files)
    
    # Copy metadata from one of the source files
    out_meta = src_files[0].meta.copy()
    
    # Update metadata for the merged raster
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform
    })
    
    # Write the merged raster to disk
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)

    nodata_val = out_meta.get("nodata", None)
    
    # Close all source files
    for src in src_files:
        src.close()

    
    #Sprint(f"Merged raster saved at: {output_file}")
