#!/usr/bin/env python3
"""
extract_replaynavs_context.py  —  Script A

Randomly samples --n_context images (default 256) from the 266 canonical
camera images in cam_imags/images/, runs them through VGGT, and saves:

  <out_dir>/context_features.pth      — (N, 5, D) tensor
  <out_dir>/context_selection.json    — manifest: which images were picked + VGGT poses

selection.json structure:
  {
    "n_context": 256,
    "feature_file": "context_features.pth",
    "frames": [
      {
        "image_id":  "SC-1025_4",
        "file_path": "/abs/path/to/SC-1025_4.jpg",
        "scene_id":  "SC-1025",
        "cam_id":    "4",
        "pose_enc":  [tx, ty, tz, qw, qx, qy, qz, fov_h, fov_w]
      },
      ...
    ]
  }
"""

import json
import random
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
def extract_tokens_and_pose(model: VGGT, imgs: torch.Tensor, device: str):
    """
    Args:
        imgs : (N, 3, H, W)  preprocessed images

    Returns:
        tokens   : (N, 5, D)  camera + register tokens (last aggregator layer)
        pose_enc : (N, 9)     VGGT-predicted poses [tx,ty,tz,qw,qx,qy,qz,fov_h,fov_w]
    """
    model.eval()
    x = imgs.to(device).unsqueeze(0)               # (1, N, 3, H, W)

    aggregated_tokens_list, _ = model.aggregator(x)

    last   = aggregated_tokens_list[-1]             # (1, N, P, D)
    tokens = last[:, :, :5, :].squeeze(0).cpu()    # (N, 5, D)

    with torch.cuda.amp.autocast(enabled=False):
        pose_enc_list = model.camera_head(aggregated_tokens_list)
        pose_enc = pose_enc_list[-1]                # (1, N, 9)
    pose_enc = pose_enc.squeeze(0).cpu()            # (N, 9)

    return tokens, pose_enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir",   required=True,
                    help="Path to cam_imags/images/ containing the 266 .jpg files")
    ap.add_argument("--out_dir",      required=True,
                    help="Directory where outputs will be written")
    ap.add_argument("--vggt_hf_repo", required=True,
                    help="HuggingFace repo ID e.g. facebook/VGGT-1B")
    ap.add_argument("--device",       default="cuda",
                    help="Torch device e.g. cuda:0")
    ap.add_argument("--n_context",    type=int, default=256,
                    help="Number of images to sample as context (default 256)")
    ap.add_argument("--seed",         type=int, default=42)
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

    # ── random sample ─────────────────────────────────────────────────────────
    rng = random.Random(args.seed)
    if len(all_images) < args.n_context:
        print(f"[warn] only {len(all_images)} images — sampling with replacement")
        selected = rng.choices(all_images, k=args.n_context)
    else:
        selected = rng.sample(all_images, k=args.n_context)

    print(f"Sampled {len(selected)} images for context (seed={args.seed})")

    # ── load model ────────────────────────────────────────────────────────────
    print(f"Loading VGGT from {args.vggt_hf_repo} → {args.device}")
    model = VGGT.from_pretrained(args.vggt_hf_repo).to(args.device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    # ── preprocess ────────────────────────────────────────────────────────────
    print("Preprocessing images ...")
    imgs = load_and_preprocess_images(
        [str(p) for p in selected], mode="pad"
    )  # (N, 3, H, W)
    print(f"Image tensor shape: {tuple(imgs.shape)}")

    # ── VGGT forward ──────────────────────────────────────────────────────────
    print("Running VGGT ...")
    tokens, pose_enc = extract_tokens_and_pose(model, imgs, args.device)
    print(f"Tokens   : {tuple(tokens.shape)}")
    print(f"Pose enc : {tuple(pose_enc.shape)}")

    # ── save features ─────────────────────────────────────────────────────────
    feat_path = out_dir / "context_features.pth"
    torch.save(tokens, feat_path)
    print(f"Saved {feat_path}")

    # ── save manifest ─────────────────────────────────────────────────────────
    manifest = {
        "n_context": len(selected),
        "feature_file": "context_features.pth",
        "frames": [
            {
                "image_id":  p.stem,                         # e.g. "SC-1025_4"
                "file_path": str(p),
                "scene_id":  p.stem.rsplit("_", 1)[0],       # e.g. "SC-1025"
                "cam_id":    p.stem.rsplit("_", 1)[1],       # e.g. "4"
                "pose_enc":  pose_enc[i].tolist(),
            }
            for i, p in enumerate(selected)
        ],
    }

    sel_path = out_dir / "context_selection.json"
    with open(sel_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {sel_path}  ({len(manifest['frames'])} entries)")


if __name__ == "__main__":
    main()
