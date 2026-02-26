from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import laspy
import numpy as np
import pdal

try:
    import open3d as o3d
except Exception:  # pragma: no cover
    o3d = None


def read_xyz_laspy(laz_path: Path) -> np.ndarray:
    las = laspy.read(str(laz_path))
    return np.vstack((las.x, las.y, las.z)).T.astype(np.float64)


def to_o3d(xyz: np.ndarray):
    if o3d is None:
        raise RuntimeError("open3d is not available. Install open3d to run ICP alignment.")
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    return pc


def run_icp(source_xyz: np.ndarray, target_xyz: np.ndarray, voxel_size: float, max_dist: float, max_iters: int):
    # Centre both clouds for numerical stability, then un-centre the returned transform.
    c_src = np.mean(source_xyz, axis=0)
    c_tgt = np.mean(target_xyz, axis=0)
    src_centered = source_xyz - c_src
    tgt_centered = target_xyz - c_tgt

    src = to_o3d(src_centered)
    tgt = to_o3d(tgt_centered)

    src = src.voxel_down_sample(float(voxel_size))
    tgt = tgt.voxel_down_sample(float(voxel_size))

    normal_radius = 3.0 * float(voxel_size)
    src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    tgt.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))

    reg = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        float(max_dist),
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iters)),
    )
    t_prime = np.asarray(reg.transformation)

    t_uncentre_tgt = np.eye(4)
    t_uncentre_tgt[:3, 3] = c_tgt
    t_uncentre_src = np.eye(4)
    t_uncentre_src[:3, 3] = -c_src
    t_world = t_uncentre_tgt @ t_prime @ t_uncentre_src
    return reg, t_world


def compute_overlap_bbox_from_headers(ref_laz: Path, src_laz: Path, buffer_m: float) -> Optional[tuple[float, float, float, float]]:
    with laspy.open(str(ref_laz)) as ref, laspy.open(str(src_laz)) as src:
        rminx, rminy = float(ref.header.min[0]), float(ref.header.min[1])
        rmaxx, rmaxy = float(ref.header.max[0]), float(ref.header.max[1])
        sminx, sminy = float(src.header.min[0]), float(src.header.min[1])
        smaxx, smaxy = float(src.header.max[0]), float(src.header.max[1])

    minx = max(rminx, sminx)
    miny = max(rminy, sminy)
    maxx = min(rmaxx, smaxx)
    maxy = min(rmaxy, smaxy)
    if maxx <= minx or maxy <= miny:
        return None

    b = float(buffer_m)
    return (minx - b, miny - b, maxx + b, maxy + b)


def _verify_output(path: Path, min_bytes: int = 10_000) -> None:
    if not path.exists():
        print(f"[WARNING] Expected output does not exist: {path}")
        return
    size = path.stat().st_size
    if size < min_bytes:
        print(f"[WARNING] Output is tiny ({size} bytes): {path}")


def crop_bbox_with_pdal(in_laz: Path, out_laz: Path, bbox: tuple[float, float, float, float]) -> None:
    out_laz.parent.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = bbox
    bounds = f"([{minx},{maxx}],[{miny},{maxy}])"
    pipeline = [
        {"type": "readers.las", "filename": str(in_laz)},
        {"type": "filters.crop", "bounds": bounds},
        {"type": "writers.las", "filename": str(out_laz), "compression": "laszip"},
    ]
    pdal.pipeline.Pipeline(json.dumps(pipeline)).execute()
    _verify_output(out_laz)


def apply_transformation_with_pdal(in_laz: Path, out_laz: Path, T: np.ndarray) -> None:
    out_laz.parent.mkdir(parents=True, exist_ok=True)
    mat = " ".join(str(float(v)) for v in T.reshape(-1))
    pipeline = [
        {"type": "readers.las", "filename": str(in_laz)},
        {"type": "filters.transformation", "matrix": mat},
        {"type": "writers.las", "filename": str(out_laz), "compression": "laszip"},
    ]
    pdal.pipeline.Pipeline(json.dumps(pipeline)).execute()
    _verify_output(out_laz)


def merge_two_laz_with_pdal(a: Path, b: Path, out_laz: Path) -> None:
    out_laz.parent.mkdir(parents=True, exist_ok=True)
    pipeline = [
        {"type": "readers.las", "filename": str(a)},
        {"type": "readers.las", "filename": str(b)},
        {"type": "filters.merge"},
        {"type": "writers.las", "filename": str(out_laz), "compression": "laszip"},
    ]
    pdal.pipeline.Pipeline(json.dumps(pipeline)).execute()
    _verify_output(out_laz)


def metrics_from_T(T: np.ndarray) -> dict:
    dx, dy, dz = float(T[0, 3]), float(T[1, 3]), float(T[2, 3])
    R = T[:3, :3]
    trace = float(np.trace(R))
    c = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    angle_deg = math.degrees(math.acos(c))
    return {"dx": dx, "dy": dy, "dz": dz, "rotation_deg": angle_deg}


def apply_transform_xyz(xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    if xyz.size == 0:
        return xyz
    ones = np.ones((xyz.shape[0], 1), dtype=np.float64)
    hom = np.hstack([xyz, ones])
    out = (T @ hom.T).T
    return out[:, :3]


def evaluate_dz_consistency(dz_from_transform: float, dz_median_pre: Optional[float], threshold: float) -> bool:
    if dz_median_pre is None:
        return True
    return abs(float(dz_from_transform) - float(dz_median_pre)) <= float(threshold)


def write_icp_report_txt(report_path: Path, step_id: int, ref_path: Path, src_path: Path,
                         params: dict, reg, T: np.ndarray, runtime_sec: float,
                         accepted: bool, reason: Optional[str], overlap_bbox: Optional[tuple]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    m = metrics_from_T(T)
    lines = [
        f"Timestamp: {datetime.now().isoformat()}",
        f"Step: {step_id}",
        f"Reference: {ref_path}",
        f"Source: {src_path}",
        f"Overlap bbox: {overlap_bbox}",
        f"Accepted: {accepted}",
        f"Reason: {reason}",
        f"Runtime (s): {runtime_sec:.3f}",
        f"Fitness: {getattr(reg, 'fitness', None)}",
        f"Inlier RMSE: {getattr(reg, 'inlier_rmse', None)}",
        f"dx/dy/dz: {m['dx']:.6f}, {m['dy']:.6f}, {m['dz']:.6f}",
        f"rotation_deg: {m['rotation_deg']:.6f}",
        f"Params: {json.dumps(params, indent=2)}",
        "Transformation (4x4):",
        np.array2string(T, precision=9, suppress_small=False),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def append_json_ledger(ledger_path: Path, record: dict) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    if ledger_path.exists():
        try:
            payload = json.loads(ledger_path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                payload = []
        except Exception:
            payload = []
    else:
        payload = []
    payload.append(record)
    ledger_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_open3d_info() -> dict:
    available = o3d is not None
    version = getattr(o3d, "__version__", None) if available else None
    return {"open3d_available": available, "open3d_version": version}


def _crs_to_dict(crs) -> dict:
    if crs is None:
        return {"present": False, "epsg": None, "wkt": None, "horizontal": None, "vertical": None, "units": None}
    epsg = None
    try:
        epsg = crs.to_epsg()
    except Exception:
        epsg = None
    axis_units = []
    try:
        axis_units = [a.unit_name for a in crs.axis_info] if crs.axis_info else []
    except Exception:
        axis_units = []
    horizontal = None
    vertical = None
    try:
        horizontal = crs.source_crs.to_wkt() if getattr(crs, "source_crs", None) else crs.to_wkt()
    except Exception:
        horizontal = None
    try:
        vertical = crs.target_crs.to_wkt() if getattr(crs, "target_crs", None) else None
    except Exception:
        vertical = None
    wkt = None
    try:
        wkt = crs.to_wkt()
    except Exception:
        wkt = None
    return {
        "present": True,
        "epsg": epsg,
        "wkt": wkt,
        "horizontal": horizontal,
        "vertical": vertical,
        "units": axis_units,
    }


def extract_las_crs_info(laz_path: Path) -> dict:
    with laspy.open(str(laz_path)) as f:
        crs = None
        try:
            crs = f.header.parse_crs()
        except Exception:
            crs = None
    return _crs_to_dict(crs)


def compute_crop_diagnostics(xyz_ref: np.ndarray, xyz_src: np.ndarray) -> dict:
    def _bounds(x):
        return {
            "min": x.min(axis=0).tolist() if len(x) else None,
            "max": x.max(axis=0).tolist() if len(x) else None,
        }
    centroid_ref = np.median(xyz_ref[:, :2], axis=0) if len(xyz_ref) else np.array([np.nan, np.nan])
    centroid_src = np.median(xyz_src[:, :2], axis=0) if len(xyz_src) else np.array([np.nan, np.nan])
    dxy = float(np.linalg.norm(centroid_ref - centroid_src)) if len(xyz_ref) and len(xyz_src) else None
    medz_ref = float(np.median(xyz_ref[:, 2])) if len(xyz_ref) else None
    medz_src = float(np.median(xyz_src[:, 2])) if len(xyz_src) else None
    dz = (medz_src - medz_ref) if (medz_ref is not None and medz_src is not None) else None
    return {
        "n_ref_crop": int(len(xyz_ref)),
        "n_src_crop": int(len(xyz_src)),
        "bounds_ref_crop": _bounds(xyz_ref),
        "bounds_src_crop": _bounds(xyz_src),
        "centroid_ref_crop": np.mean(xyz_ref, axis=0).tolist() if len(xyz_ref) else None,
        "centroid_src_crop": np.mean(xyz_src, axis=0).tolist() if len(xyz_src) else None,
        "median_z_ref_crop": medz_ref,
        "median_z_src_crop": medz_src,
        "dxy_centroid_pre": dxy,
        "dz_median_pre": dz,
    }


def bounds_xy_from_xyz(xyz: np.ndarray) -> Optional[tuple[float, float, float, float]]:
    if xyz.size == 0:
        return None
    return (
        float(np.min(xyz[:, 0])),
        float(np.min(xyz[:, 1])),
        float(np.max(xyz[:, 0])),
        float(np.max(xyz[:, 1])),
    )


def intersect_bboxes(b1: tuple[float, float, float, float], b2: tuple[float, float, float, float]) -> Optional[tuple[float, float, float, float]]:
    xmin = max(float(b1[0]), float(b2[0]))
    ymin = max(float(b1[1]), float(b2[1]))
    xmax = min(float(b1[2]), float(b2[2]))
    ymax = min(float(b1[3]), float(b2[3]))
    if xmin >= xmax or ymin >= ymax:
        return None
    return (xmin, ymin, xmax, ymax)


def buffer_bbox_inward(bbox: tuple[float, float, float, float], buffer_m: float) -> Optional[tuple[float, float, float, float]]:
    b = float(buffer_m)
    xmin, ymin, xmax, ymax = bbox
    out = (xmin + b, ymin + b, xmax - b, ymax - b)
    if out[0] >= out[2] or out[1] >= out[3]:
        return None
    return out


def bbox_area_m2(bbox: tuple[float, float, float, float]) -> float:
    return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def host_info() -> dict:
    return {
        "hostname": socket.gethostname(),
        "sys_executable": sys.executable,
        "python_version": sys.version,
    }


def _main():
    parser = argparse.ArgumentParser(description="Run overlap-based ICP alignment between two LAZ files.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--voxel_size", type=float, default=1.0)
    parser.add_argument("--max_dist", type=float, default=2.0)
    parser.add_argument("--max_iters", type=int, default=80)
    parser.add_argument("--overlap_buffer_m", type=float, default=10.0)
    parser.add_argument("--disable_overlap_crop", action="store_true")
    args = parser.parse_args()

    src = Path(args.source)
    ref = Path(args.reference)
    out = Path(args.out)
    report = Path(args.report)

    work_dir = report.parent / "tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    overlap = compute_overlap_bbox_from_headers(ref, src, args.overlap_buffer_m)
    ref_icp = ref
    src_icp = src
    if overlap and not args.disable_overlap_crop:
        ref_icp = work_dir / "ref_subset.laz"
        src_icp = work_dir / "src_subset.laz"
        crop_bbox_with_pdal(ref, ref_icp, overlap)
        crop_bbox_with_pdal(src, src_icp, overlap)

    t0 = time.time()
    reg, T = run_icp(
        source_xyz=read_xyz_laspy(src_icp),
        target_xyz=read_xyz_laspy(ref_icp),
        voxel_size=args.voxel_size,
        max_dist=args.max_dist,
        max_iters=args.max_iters,
    )
    dt = time.time() - t0

    apply_transformation_with_pdal(src, out, T)
    write_icp_report_txt(
        report_path=report,
        step_id=1,
        ref_path=ref,
        src_path=src,
        params={
            "voxel_size": args.voxel_size,
            "max_dist": args.max_dist,
            "max_iters": args.max_iters,
            "overlap_buffer_m": args.overlap_buffer_m,
            "overlap_crop": (overlap is not None and not args.disable_overlap_crop),
        },
        reg=reg,
        T=T,
        runtime_sec=dt,
        accepted=True,
        reason=None,
        overlap_bbox=overlap,
    )
    print(f"Aligned source written to: {out}")
    print(f"Report written to: {report}")


if __name__ == "__main__":
    _main()
