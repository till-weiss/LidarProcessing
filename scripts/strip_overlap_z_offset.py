import json
import numpy as np
import pdal

from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


def pdal_bounds(laz_path: str):
    """Return (minx, maxx, miny, maxy) from PDAL info."""
    info = json.loads(pdal.Pipeline(json.dumps({
        "pipeline": [laz_path, {"type": "filters.info"}]
    })).execute_streaming() or "{}")  # execute_streaming returns count, not JSON

    # The Python PDAL bindings don't always return info via execute(); so use pdal info via metadata
    p = pdal.Pipeline(json.dumps({"pipeline": [laz_path]}))
    p.execute()
    md = json.loads(p.metadata)

    b = md["metadata"]["readers.las"]["bbox"]["native"]["bbox"]
    minx, maxx = b["minx"], b["maxx"]
    miny, maxy = b["miny"], b["maxy"]
    return minx, maxx, miny, maxy


def intersection_bounds(b1, b2):
    minx = max(b1[0], b2[0])
    maxx = min(b1[1], b2[1])
    miny = max(b1[2], b2[2])
    maxy = min(b1[3], b2[3])
    if (minx >= maxx) or (miny >= maxy):
        raise ValueError("No overlap found between the two strips (intersection bbox is empty).")
    return minx, maxx, miny, maxy


def read_overlap_points(laz_path: str, bounds, voxel=1.0, keep_classes=None, limit=None):
    """
    Crop to overlap bounds and return Nx3 array (X,Y,Z).
    - voxel: voxelgrid cell size (m) to thin points
    - keep_classes: e.g. [2] for ground only (if classification exists)
    - limit: random subsample cap (int) after reading (optional safety)
    """
    minx, maxx, miny, maxy = bounds
    crop_bounds = f"([{minx},{maxx}],[{miny},{maxy}])"

    pipeline = {"pipeline": [laz_path, {"type": "filters.crop", "bounds": crop_bounds}]}

    # Optional class filter (only if you trust classifications)
    if keep_classes is not None:
        cls_list = ",".join(str(c) for c in keep_classes)
        pipeline["pipeline"].append({"type": "filters.range", "limits": f"Classification[{cls_list}:{cls_list}]"})

    # Voxel thinning (fast + spatially uniform)
    if voxel is not None and voxel > 0:
        pipeline["pipeline"].append({"type": "filters.voxelgrid", "cell": float(voxel)})

    p = pdal.Pipeline(json.dumps(pipeline))
    count = p.execute()
    if count == 0:
        return np.empty((0, 3), dtype=float)

    arr = p.arrays[0]
    xyz = np.vstack([arr["X"], arr["Y"], arr["Z"]]).T.astype(np.float64)

    # Optional cap (random) to keep KDTree quick on very dense overlaps
    if limit is not None and xyz.shape[0] > limit:
        idx = np.random.default_rng(42).choice(xyz.shape[0], size=limit, replace=False)
        xyz = xyz[idx]

    return xyz


def robust_stats(dz):
    dz = dz[np.isfinite(dz)]
    med = np.median(dz)
    mad = np.median(np.abs(dz - med))
    # Convert MAD to sigma-ish (normal dist) for intuition
    mad_sigma = 1.4826 * mad

    mean = float(np.mean(dz))
    std = float(np.std(dz))

    # Trimmed mean (5% each tail) – stable against outliers
    lo, hi = np.quantile(dz, [0.05, 0.95])
    trimmed = dz[(dz >= lo) & (dz <= hi)]
    tmean = float(np.mean(trimmed))
    tstd = float(np.std(trimmed))

    return {
        "n": int(dz.size),
        "mean": mean,
        "std": std,
        "median": float(med),
        "mad": float(mad),
        "mad_sigma": float(mad_sigma),
        "trimmed_mean_5_95": tmean,
        "trimmed_std_5_95": tstd,
        "q05": float(lo),
        "q95": float(hi),
    }


def main(strip1, strip2, voxel=1.0, max_nn_dist=1.5, keep_ground_only=False, cap_points=2_000_000):
    # 1) Bounds + overlap bbox
    b1 = pdal_bounds(strip1)
    b2 = pdal_bounds(strip2)
    ib = intersection_bounds(b1, b2)
    print("Overlap bounds (minx, maxx, miny, maxy):")
    print(ib)

    # 2) Read overlap points (cropped + voxel thinned)
    keep_classes = [2] if keep_ground_only else None
    p1 = read_overlap_points(strip1, ib, voxel=voxel, keep_classes=keep_classes, limit=cap_points)
    p2 = read_overlap_points(strip2, ib, voxel=voxel, keep_classes=keep_classes, limit=cap_points)

    print(f"Strip1 overlap points: {p1.shape[0]:,}")
    print(f"Strip2 overlap points: {p2.shape[0]:,}")

    if p1.size == 0 or p2.size == 0:
        raise RuntimeError("One of the overlap point sets is empty after cropping/filtering.")

    # 3) Nearest-neighbour match in XY, compute ΔZ
    tree = cKDTree(p2[:, :2])
    dist, idx = tree.query(p1[:, :2], k=1, workers=-1)

    # Keep only matches within a sensible XY distance (prevents weird cross-matches)
    m = dist <= max_nn_dist
    if not np.any(m):
        raise RuntimeError("No nearest-neighbour matches within max_nn_dist. Increase max_nn_dist or reduce voxel size.")

    dz = p2[idx[m], 2] - p1[m, 2]  # strip2 - strip1
    stats = robust_stats(dz)

    print("\nΔZ = Z(strip2) - Z(strip1) in overlap (nearest-neighbour in XY)")
    for k, v in stats.items():
        print(f"{k:>18}: {v}")

    # 4) Plots
    # Histogram
    plt.figure()
    plt.hist(dz, bins=80)
    plt.xlabel("ΔZ (m)")
    plt.ylabel("Count")
    plt.title("Overlap ΔZ histogram (strip2 - strip1)")
    plt.tight_layout()
    plt.savefig("dz_hist.png", dpi=200)
    print("\nSaved histogram: dz_hist.png")

    # Optional: spatial view (downsample more for plot)
    if dz.size > 300_000:
        sel = np.random.default_rng(42).choice(dz.size, size=300_000, replace=False)
        dz_plot = dz[sel]
        xy_plot = p1[m, :2][sel]
    else:
        dz_plot = dz
        xy_plot = p1[m, :2]

    plt.figure()
    plt.scatter(xy_plot[:, 0], xy_plot[:, 1], s=1, c=dz_plot)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Overlap ΔZ spatial pattern (colour = ΔZ)")
    plt.tight_layout()
    plt.savefig("dz_map.png", dpi=200)
    print("Saved map: dz_map.png")

    # 5) Suggested correction (if you want to shift strip2 to match strip1)
    print("\nIf you want to vertically align strip2 to strip1, a robust shift is:")
    print(f"  dz_shift = -median = {-stats['median']:.4f} m  (apply to strip2)")


if __name__ == "__main__":
    # ---- EDIT THESE PATHS ----
    strip1 = "/Volumes/Till_SSD/Lidar_Peel_2023/dsampled/FULL_ALS_L1B_20230707T152300_153124_voxel_5m.laz"
    strip2 = "/Volumes/Till_SSD/Lidar_Peel_2023/dsampled/FULL_ALS_L1B_20230707T154040_154718_voxel_1m.laz"

    # Parameters you might tweak:
    # voxel: 0.5 or 1.0 are good starts
    # max_nn_dist: should be ~1–2x voxel size
    main(
        strip1,
        strip2,
        voxel=1.0,
        max_nn_dist=1.5,
        keep_ground_only=False,  # set True if classification=2 is trustworthy
        cap_points=2_000_000
    )
