"""
Cache Mid (M) Signal Per Context Frame for ReplayNVAS
====================================================

For each image in context_selection.json, loads the corresponding 1s .wav file
from ReplayNVAS/v3/<scene_id>/0/<cam_id>.wav and computes M = (L + R) / 2.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import librosa
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default="ReplayNVAS/replaynavs_outputs_256/context_selection.json")
    parser.add_argument("--v3_dir", default="ReplayNVAS/v3")
    parser.add_argument("--output", default="ReplayNVAS/replaynavs_outputs_256/mid_signal_cache.json")
    parser.add_argument("--sr", default=22050, type=int)
    args = parser.parse_args()

    print(f"Loading selection from: {args.selection}")
    with open(args.selection, "r") as f:
        selection = json.load(f)

    v3_path = Path(args.v3_dir)
    mid_cache = {}

    # Structure in context_selection.json:
    # { "frames": [ { "scene_id": "SC-1042", "cam_id": "4", "image_id": "SC-1042_4", ... }, ... ] }
    
    frames = selection["frames"]
    print(f"Found {len(frames)} frames to process.")

    for frame_info in tqdm(frames, desc="Processing frames"):
        scene_id = frame_info["scene_id"]
        cam_id = frame_info["cam_id"]
        image_id = frame_info["image_id"]
        
        # In ReplayNVAS/v3, audio files are named <cam_id>.wav inside <scene_id>/0/ for block 0
        # Check if it exists.
        audio_path = v3_path / scene_id / "0" / f"{cam_id}.wav"
        
        if not audio_path.exists():
            # Try finding it in any block if 0 doesn't exist? 
            # But context images are usually the first ones.
            # Let's check block 0 first.
            print(f"\n[WARN] {audio_path} not found. Searching other blocks...")
            blocks = sorted([d.name for d in (v3_path / scene_id).iterdir() if d.is_dir() and d.name.isdigit()])
            found = False
            for b in blocks:
                alt_path = v3_path / scene_id / b / f"{cam_id}.wav"
                if alt_path.exists():
                    audio_path = alt_path
                    found = True
                    break
            if not found:
                print(f"[ERROR] Could not find any .wav for {scene_id} cam {cam_id}")
                continue

        try:
            # Load audio (force sr, mono=False to get stereo if available)
            y, sr = librosa.load(str(audio_path), sr=args.sr, mono=False)
            
            if y.ndim == 2:
                # Stereo: M = (L + R) / 2
                m = (y[0] + y[1]) / 2.0
            else:
                # Mono: M = y
                m = y
            
            # Pad or truncate to exact sr (1 second)
            if len(m) < args.sr:
                m = np.pad(m, (0, args.sr - len(m)))
            elif len(m) > args.sr:
                m = m[:args.sr]
                
            if scene_id not in mid_cache:
                mid_cache[scene_id] = {}
            
            # Use image_id as key to be safe, or just follow the same scene/frame structure?
            # RWAVS used scene_id -> frame_id. ReplayNVAS context doesn't have frame_id in the same way.
            # But the model might expect mid_cache[scene_id][image_id] or mid_cache[scene_id][frame_index]
            # Let's use image_id for now.
            mid_cache[scene_id][image_id] = m.astype(np.float32).tolist()
            
        except Exception as e:
            print(f"\n[ERROR] Failed to process {audio_path}: {e}")

    print(f"\nWriting cache to: {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(mid_cache, f)

    print("Success ✓")

if __name__ == "__main__":
    main()
