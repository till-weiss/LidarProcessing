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
<<<<<<< HEAD
        {
            "type": "filters.range",
            "limits": "Classification![7:7]"  # Use all points for initial DSM, except noise
        },
=======
       # FOr testing  {
       #     "type": "filters.range",
       #     "limits": "Classification[0:0]"  # Use all points for initial DSM
       # },
>>>>>>> 3395522 (Update)
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

    chunk_file = os.path.join(
        temp_dir,
        f"{os.path.basename(input_file).replace('.las', '')}_chunk_{int(small_chunk_bbox.bounds[0])}_{int(small_chunk_bbox.bounds[1])}.tif"
    )

    pipeline = [
        {"type": "readers.las", "filename": input_file},
        {"type": "filters.crop", "polygon": wkt_dumps(large_chunk_bbox)},

    ]
    if filter_smrf:

        pipeline.append({"type": "filters.smrf",
         "scalar": float(scalar),
         "slope": float(slope),
         "window": float(window)})
        
    if filter_csf:
        pipeline.append(
        {"type": "filters.csf",
         "resolution": float(cloth_resolution),
         "rigidness": int(rigidness),
         "iterations": int(iterations),
         "step": float(time_step)})
        
    pipeline += [
        {"type": "filters.ferry", "dimensions": "Z=>Elevation"},
        {"type": "filters.range", "limits": "Classification[2:2]"},
        {"type": "filters.crop", "polygon": wkt_dumps(small_chunk_bbox)},
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
                "-md", "10",
                "-si", "2",
                chunk_file,
                chunk_file
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] GDAL fillnodata failed: {e}")
            return None

    return chunk_file

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