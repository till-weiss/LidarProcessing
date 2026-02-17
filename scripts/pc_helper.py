from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import laspy
import open3d as o3d


def las_to_o3d(
    path: str | os.PathLike,
    *,
    voxel_size: float | None = None,
    estimate_normals: bool = False,
    normal_radius: float = 3.0,
    normal_max_nn: int = 30,
    align_normals_up: bool = True,
) -> o3d.geometry.PointCloud:
    """
    Load a .las/.laz file with laspy and convert to Open3D PointCloud.
    Optionally voxel-downsample and compute/orient normals.
    """
    path = Path(path)
    if path.suffix.lower() not in {".las", ".laz"}:
        raise ValueError(f"Expected .las or .laz, got: {path.suffix}")

    las = laspy.read(str(path))

    # XYZ (scaled coordinates)
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    if xyz.size == 0:
        raise ValueError(f"No points found in: {path}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Optional RGB
    has_rgb = all(hasattr(las, c) for c in ("red", "green", "blue"))
    if has_rgb:
        rgb = np.column_stack((las.red, las.green, las.blue)).astype(np.float64)

        # Normalise to [0,1]. Most LAS RGB is 16-bit (0..65535), but some are 8-bit (0..255).
        denom = 65535.0 if rgb.max() > 255 else 255.0
        pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb / denom, 0.0, 1.0))

    # Optional downsample
    if voxel_size is not None:
        if voxel_size <= 0:
            raise ValueError("voxel_size must be > 0")
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))

    # Optional normals
    if estimate_normals:
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=float(normal_radius), max_nn=int(normal_max_nn))
        )
        if align_normals_up:
            pcd.orient_normals_to_align_with_direction([0.0, 0.0, 1.0])

    if pcd.is_empty():
        raise ValueError(f"Point cloud is empty after processing: {path}")

    return pcd