"""Build PCA-reconstructed teacher profiles for profile-reconstruction ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


COMPONENT_SLICES = {
    "pc1": (1, 1),
    "pc1_5": (1, 5),
    "pc1_10": (1, 10),
    "pc2_10": (2, 10),
    "full": None,
}


def _finite_impute(values, center):
    values = np.asarray(values, dtype=np.float64)
    out = values.copy()
    missing = ~np.isfinite(out)
    if np.any(missing):
        out[missing] = np.take(center, np.where(missing)[1])
    return out


def _fit_transformer(train_raw):
    train_raw = np.asarray(train_raw, dtype=np.float64)
    center = np.nanmedian(train_raw, axis=0)
    center[~np.isfinite(center)] = 0.0
    train_filled = _finite_impute(train_raw, center)
    scale = np.nanstd(train_filled, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    standardized = (train_filled - center) / scale
    pca = PCA(svd_solver="full")
    pca.fit(standardized)
    return center, scale, pca


def _reconstruct(raw, center, scale, pca, component_spec):
    if component_spec is None:
        return np.asarray(raw, dtype=np.float32)
    start, end = component_spec
    filled = _finite_impute(raw, center)
    standardized = (filled - center) / scale
    scores = pca.transform(standardized)
    kept = np.zeros_like(scores)
    kept[:, start - 1 : end] = scores[:, start - 1 : end]
    reconstructed = pca.inverse_transform(kept) * scale + center
    return reconstructed.astype(np.float32)


def build_profile(source_profile_dir, output_profile_dir, component_set):
    source_profile_dir = Path(source_profile_dir)
    output_profile_dir = Path(output_profile_dir)
    arrays = np.load(source_profile_dir / "profile_arrays.npz", allow_pickle=True)
    train_raw = arrays["train_oof_raw_score"]
    center, scale, pca = _fit_transformer(train_raw)
    spec = COMPONENT_SLICES[component_set]

    payload = {}
    for key in arrays.files:
        if key.endswith("_raw_score"):
            payload[key] = _reconstruct(arrays[key], center, scale, pca, spec)
        else:
            payload[key] = arrays[key]

    output_profile_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_profile_dir / "profile_arrays.npz", **payload)
    for key, value in payload.items():
        if key.endswith("_raw_score") or key.endswith("_y") or key.endswith("_row_index"):
            np.save(output_profile_dir / f"{key}.npy", value)
    metadata = {
        "source_profile_dir": str(source_profile_dir),
        "component_set": component_set,
        "component_range_1_indexed": COMPONENT_SLICES[component_set],
        "fit_rows": int(train_raw.shape[0]),
        "fit_dimensions": int(train_raw.shape[1]),
        "standardization": "training OOF median center and standard deviation",
        "construction": "inverse PCA reconstruction after zeroing unselected components",
    }
    (output_profile_dir / "profile_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_profile_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Build PCA-truncated teacher profiles for soft supervision.")
    parser.add_argument("--source_profile_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--component_sets", default="pc1,pc1_5,pc1_10,pc2_10")
    parser.add_argument("--profile_prefix", default="feature=pca_reconstruction__model=Teacher__split=pca_soft__seed=mix")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    for component_set in [item.strip() for item in args.component_sets.split(",") if item.strip()]:
        if component_set not in COMPONENT_SLICES:
            raise ValueError(f"Unknown component set: {component_set}")
        output = output_root / f"{args.profile_prefix}__components={component_set}"
        print(f"Building {component_set} -> {output}")
        build_profile(args.source_profile_dir, output, component_set)


if __name__ == "__main__":
    main()
