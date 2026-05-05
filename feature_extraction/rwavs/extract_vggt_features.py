#!/usr/bin/env python3
"""
extract_vggt_features_v3.py

Per-scene extraction (not per-target-frame).

For each scene:
  1. Collect all images from <data_root>/<scene_id>/frames/
  2. Randomly select 256 (without replacement; with replacement if <256 frames)
  3. Preprocess using VGGT's official load_and_preprocess_images (mode="pad", target 518px)
  4. Pass all 256 through VGGT at once -> (256, 5, D) tokens + (256, 9) poses from camera_head
  5. Save one .pth per scene

One single selection.json at <out_root>/selection.json covers all 13 scenes.

Output layout:
  <out_root>/
    <scene_id>/features.pth      # tensor (256, 5, D)
    selection.json               # manifest for all scenes

selection.json structure:
  {
    "<scene_id>": {
      "scene_id": "<scene_id>",
      "feature_file": "<scene_id>/features.pth",
      "frames": [
        {
          "frame_id": "00001",
          "file_path": "frames/00001.png",
          "pose_enc": [tx, ty, tz, qw, qx, qy, qz, fov_h, fov_w]
        },
        ... 256 entries
      ]
    },
    ...
  }

pose_enc layout: [tx, ty, tz, qw, qx, qy, qz, fov_h, fov_w]  (from VGGT camera_head)
"""

import json
import random
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch

# Add vggt directory to sys.path
vggt_path = Path(__file__).resolve().parent.parent / "vggt"
if vggt_path.exists():
    sys.path.append(str(vggt_path))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        return json.load(f)

def save_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)


@torch.no_grad()
def extract_tokens_and_pose(vggt_model: VGGT, imgs: torch.Tensor, device: str):
    """
    Args:
        imgs: (N, 3, H, W)  — preprocessed images (output of load_and_preprocess_images)

    Returns:
        tokens:   (N, 5, D)   — camera + register tokens from last aggregator layer
        pose_enc: (N, 9)      — VGGT predicted pose per image
                                layout: [tx, ty, tz, qw, qx, qy, qz, fov_h, fov_w]
    """
    vggt_model.eval()
    x = imgs.to(device).unsqueeze(0)              # (1, N, 3, H, W)

    aggregated_tokens_list, _ = vggt_model.aggregator(x)

    last = aggregated_tokens_list[-1]              # (1, N, P, D)
    tokens = last[:, :, :5, :].squeeze(0).cpu()   # (N, 5, D)

    with torch.cuda.amp.autocast(enabled=False):
        pose_enc_list = vggt_model.camera_head(aggregated_tokens_list)
        pose_enc = pose_enc_list[-1]               # (1, N, 9)
    pose_enc = pose_enc.squeeze(0).cpu()           # (N, 9)

    return tokens, pose_enc


def collect_frames(frames_dir: Path) -> List[Path]:
    """Return sorted list of all image paths in frames_dir."""
    return sorted([p for p in frames_dir.iterdir() if p.suffix in IMAGE_EXTENSIONS])


def process_scene(
    scene_id: str,
    data_root: Path,
    out_root: Path,
    vggt_model: VGGT,
    device: str,
    n_frames: int,
    seed: int,
) -> Optional[Dict[str, Any]]:
    frames_dir = data_root / scene_id / "frames"
    if not frames_dir.exists():
        print(f"[skip] frames dir not found: {frames_dir}")
        return None

    all_paths = collect_frames(frames_dir)
    if len(all_paths) == 0:
        print(f"[skip] no images found in {frames_dir}")
        return None

    rng = random.Random(seed)
    if len(all_paths) < n_frames:
        print(f"[warn] scene {scene_id} has only {len(all_paths)} frames — sampling with replacement")
        selected_paths = rng.choices(all_paths, k=n_frames)
    else:
        selected_paths = rng.sample(all_paths, k=n_frames)

    print(f"[scene {scene_id}] loading and preprocessing {n_frames} images via VGGT loader ...")
    # Use VGGT's own preprocessing: preserves aspect ratio, target 518px, divisible by 14
    imgs = load_and_preprocess_images(
        [str(p) for p in selected_paths], mode="pad"
    )  # (N, 3, H, W)

    print(f"[scene {scene_id}] image tensor shape: {tuple(imgs.shape)}")
    print(f"[scene {scene_id}] running VGGT ...")
    tokens, pose_enc = extract_tokens_and_pose(vggt_model, imgs, device)
    # tokens:   (N, 5, D)
    # pose_enc: (N, 9)

    feat_dir = out_root / scene_id
    feat_dir.mkdir(parents=True, exist_ok=True)
    feat_path = feat_dir / "features.pth"
    torch.save(tokens, feat_path)
    print(f"[scene {scene_id}] saved {feat_path}  shape={tuple(tokens.shape)}")

    scene_entry = {
        "scene_id": scene_id,
        "feature_file": str(feat_path.relative_to(out_root)),
        "frames": [
            {
                "frame_id": p.stem,
                "file_path": str(p.relative_to(data_root / scene_id)),
                "pose_enc": pose_enc[i].tolist(),
            }
            for i, p in enumerate(selected_paths)
        ],
    }

    return scene_entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",    required=True,  help="Root containing scene subdirectories (1-13)")
    ap.add_argument("--out_root",     required=True,  help="Root folder where outputs will be written")
    ap.add_argument("--scene_ids",    nargs="+", required=True, help="Scene IDs to process e.g. 1 2 3")
    ap.add_argument("--vggt_hf_repo", required=True,  help="HuggingFace repo ID e.g. facebook/VGGT-1B")
    ap.add_argument("--device",       default="cuda", help="Torch device e.g. cuda:0")
    ap.add_argument("--n_frames",     type=int, default=256, help="Number of frames to randomly select per scene")
    ap.add_argument("--seed",         type=int, default=0)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root  = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    selection_path = out_root / "selection.json"
    if selection_path.exists():
        selection = load_json(selection_path)
        print(f"Resuming — {len(selection)} scene(s) already done")
    else:
        selection = {}

    vggt_model = VGGT.from_pretrained(args.vggt_hf_repo).to(args.device)
    for p in vggt_model.parameters():
        p.requires_grad_(False)
    vggt_model.eval()

    for sid in args.scene_ids:
        if sid in selection and (out_root / sid / "features.pth").exists():
            print(f"[skip] scene {sid} already complete")
            continue

        scene_entry = process_scene(
            scene_id=sid,
            data_root=data_root,
            out_root=out_root,
            vggt_model=vggt_model,
            device=args.device,
            n_frames=args.n_frames,
            seed=args.seed + int(sid),
        )

        if scene_entry is not None:
            selection[sid] = scene_entry
            save_json(selection_path, selection)
            print(f"[scene {sid}] manifest updated → {selection_path}")


if __name__ == "__main__":
    main()
