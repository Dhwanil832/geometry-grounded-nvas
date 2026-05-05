#!/usr/bin/env python3
"""
extract_replaynavs_targets.py  —  Script B

Runs ALL 266 canonical camera images through VGGT and saves a pose
lookup dictionary so the dataloader can retrieve any target's pose by image_id.

Output:
  <out_dir>/target_poses.json   — {"SC-1025_4": [tx,ty,tz,qw,qx,qy,qz,fov_h,fov_w], ...}

Note: poses here are VGGT-predicted (same network as Script A), so they live
in the exact same coordinate space as the context poses — no calibration gap.
"""

import json
import argparse
import sys
from pathlib import Path

import torch

# ── resolve vggt package ──────────────────────────────────────────────────────
vggt_path = Path(__file__).resolve().parent.parent / "vggt"
if vggt_path.exists():
    sys.path.append(str(vggt_path))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


@torch.no_grad()
def extract_poses(model: VGGT, imgs: torch.Tensor, device: str) -> torch.Tensor:
    """
    Args:
        imgs : (N, 3, H, W)  preprocessed images

    Returns:
        pose_enc : (N, 9)  VGGT-predicted poses [tx,ty,tz,qw,qx,qy,qz,fov_h,fov_w]
    """
    model.eval()
    x = imgs.to(device).unsqueeze(0)               # (1, N, 3, H, W)

    aggregated_tokens_list, _ = model.aggregator(x)

    with torch.cuda.amp.autocast(enabled=False):
        pose_enc_list = model.camera_head(aggregated_tokens_list)
        pose_enc = pose_enc_list[-1]               # (1, N, 9)

    return pose_enc.squeeze(0).cpu()               # (N, 9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir",   required=True,
                    help="Path to cam_imags/images/ containing the 266 .jpg files")
    ap.add_argument("--out_dir",      required=True,
                    help="Directory where target_poses.json will be written")
    ap.add_argument("--vggt_hf_repo", required=True,
                    help="HuggingFace repo ID e.g. facebook/VGGT-1B")
    ap.add_argument("--device",       default="cuda",
                    help="Torch device e.g. cuda:1")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── collect all canonical camera images ───────────────────────────────────
    all_images = sorted([
        p for p in images_dir.iterdir()
        if p.suffix in IMAGE_EXTENSIONS
    ])
    print(f"Found {len(all_images)} images in {images_dir}")

    # ── load model ────────────────────────────────────────────────────────────
    print(f"Loading VGGT from {args.vggt_hf_repo} → {args.device}")
    model = VGGT.from_pretrained(args.vggt_hf_repo).to(args.device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    # ── preprocess ────────────────────────────────────────────────────────────
    print("Preprocessing images ...")
    imgs = load_and_preprocess_images(
        [str(p) for p in all_images], mode="pad"
    )  # (N, 3, H, W)
    print(f"Image tensor shape: {tuple(imgs.shape)}")

    # ── VGGT forward (poses only) ─────────────────────────────────────────────
    print("Running VGGT ...")
    pose_enc = extract_poses(model, imgs, args.device)
    print(f"Pose enc : {tuple(pose_enc.shape)}")

    # ── save pose lookup ──────────────────────────────────────────────────────
    target_poses = {
        p.stem: pose_enc[i].tolist()          # "SC-1025_4" → [9 floats]
        for i, p in enumerate(all_images)
    }

    out_path = out_dir / "target_poses.json"
    with open(out_path, "w") as f:
        json.dump(target_poses, f, indent=2)
    print(f"Saved {out_path}  ({len(target_poses)} entries)")


if __name__ == "__main__":
    main()
