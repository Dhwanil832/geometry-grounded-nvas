"""
Cache Mid (M) Signal Per Context Frame
=======================================

For each scene in selection.json, computes M = (L + R) / 2 for every context
frame using the ground-truth binaural audio, then writes the result to:

    RWAVS/vggt_outputs_v3/mid_signal_cache.json

The M signal is the raw waveform (22050 samples per frame at 22050 Hz = 1 s),
identical in nature to source_gt_audio fed into the model forward pass.

Math derivation:
    Left  = M(1 - diff)
    Right = M(1 + diff)
    Left + Right = 2M  →  M = (Left + Right) / 2

Usage:
    python tools/cache_mid_signal.py
         [--selection  path/to/selection.json]
         [--release    path/to/RWAVS/release]
         [--output     path/to/mid_signal_cache.json]
         [--sr         22050]
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Defaults (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEFAULT_SELECTION = os.path.join(PROJECT_ROOT, "RWAVS", "vggt_outputs_v3", "selection.json")
DEFAULT_RELEASE   = os.path.join(PROJECT_ROOT, "RWAVS", "release")
DEFAULT_OUTPUT    = os.path.join(PROJECT_ROOT, "RWAVS", "vggt_outputs_v3", "mid_signal_cache.json")
DEFAULT_SR        = 22050   # samples per frame (1 s × 22050 Hz)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_audio_pkl(pkl_path: str):
    """Load a 22050_audio_{split}_48k.pkl and return the (2, N) numpy array."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data   # dict: { scene_id_str: {"gt": ndarray(2, N), "source": ...} }


def get_scene_audio(release_root: str, scene_id: str, sr: int):
    """
    Merge train and val PKL audio for a scene.

    Both PKLs contain the *full* binaural array (the whole recording), so we
    just pick whichever is available.  If both exist, use train (they are the
    same recording — only the frame-index split differs between train/val).

    Returns: ndarray of shape (2, total_samples)
    """
    for split in ("train", "val"):
        pkl_path = os.path.join(release_root, scene_id,
                                f"22050_audio_{split}_48k.pkl")
        if os.path.exists(pkl_path):
            audio_data = load_audio_pkl(pkl_path)
            if scene_id in audio_data:
                return audio_data[scene_id]["gt"]  # (2, N)
    raise FileNotFoundError(
        f"Could not find any audio PKL for scene {scene_id} under {release_root}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Cache M=(L+R)/2 per context frame")
    parser.add_argument("--selection", default=DEFAULT_SELECTION,
                        help="Path to selection.json")
    parser.add_argument("--release",   default=DEFAULT_RELEASE,
                        help="Path to RWAVS/release directory")
    parser.add_argument("--output",    default=DEFAULT_OUTPUT,
                        help="Output JSON path")
    parser.add_argument("--sr",        default=DEFAULT_SR, type=int,
                        help="Samples per frame (= sample rate for 1-s frames)")
    return parser.parse_args()


def main():
    args = parse_args()
    sr = args.sr

    # --- Load selection.json ---
    print(f"Loading selection from: {args.selection}")
    with open(args.selection, "r") as f:
        selection = json.load(f)

    total_scenes = len(selection)
    print(f"Found {total_scenes} scene(s) in selection.json\n")

    mid_cache = {}   # scene_id -> { frame_id -> [m_samples] }

    for scene_id, scene_data in tqdm(selection.items(), desc="Scenes"):
        frames = scene_data["frames"]   # list of 256 dicts with frame_id

        # --- Load GT binaural audio for this scene ---
        try:
            gt_audio = get_scene_audio(args.release, scene_id, sr)
            # gt_audio: (2, N)  — channel 0 = Left, channel 1 = Right
        except FileNotFoundError as e:
            print(f"\n[WARN] Skipping scene {scene_id}: {e}")
            continue

        total_samples = gt_audio.shape[1]

        mid_cache[scene_id] = {}

        for frame_info in frames:
            frame_id = frame_info["frame_id"]   # e.g. "00138"
            frame_int = int(frame_id)            # e.g. 138

            start = (frame_int - 1) * sr
            end   = frame_int * sr

            # Guard against audio shorter than this frame
            if start >= total_samples:
                # Return zeros (same padding strategy as the dataset loader)
                m = np.zeros(sr, dtype=np.float32)
            else:
                left  = gt_audio[0, start:end]   # (sr,)
                right = gt_audio[1, start:end]   # (sr,)

                # Pad if the last frame is shorter than sr
                if left.shape[0] < sr:
                    pad = sr - left.shape[0]
                    left  = np.pad(left,  (0, pad))
                    right = np.pad(right, (0, pad))

                # M = (Left + Right) / 2  — MS stereo mid channel
                m = (left + right) / 2.0          # (sr,)

            # Store as Python list of float32-cast values
            mid_cache[scene_id][frame_id] = m.astype(np.float32).tolist()

    # --- Sanity checks ---
    print("\n--- Sanity checks ---")
    ok = True
    for scene_id, frames_dict in mid_cache.items():
        n_frames = len(frames_dict)
        expected = len(selection[scene_id]["frames"])
        if n_frames != expected:
            print(f"[FAIL] Scene {scene_id}: expected {expected} frames, got {n_frames}")
            ok = False
        for fid, m_vals in frames_dict.items():
            if len(m_vals) != sr:
                print(f"[FAIL] Scene {scene_id} frame {fid}: expected {sr} samples, got {len(m_vals)}")
                ok = False
    if ok:
        print(f"All {len(mid_cache)} scenes passed.  Each frame has {sr} M-samples.")

    # --- Write output ---
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"\nWriting cache to: {args.output}")
    with open(args.output, "w") as f:
        json.dump(mid_cache, f)

    print("Done ✓")


if __name__ == "__main__":
    main()
