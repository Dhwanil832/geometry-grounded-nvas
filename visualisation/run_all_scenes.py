import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run all visualizations for all scenes')
    parser.add_argument('--scenes', type=str, default='1-13', help='Range of scenes to process (e.g., 1-13 or 1,2,5)')
    parser.add_argument('--n_frames', type=int, default=128)
    parser.add_argument('--frame', type=int, default=98, help='Frame number for comparative plot')
    parser.add_argument('--extract', action='store_true', help='Re-extract audio for the specific frame')
    args = parser.parse_args()

    # Parse scenes
    if '-' in args.scenes:
        start, end = map(int, args.scenes.split('-'))
        scene_list = range(start, end + 1)
    else:
        scene_list = [int(s) for s in args.scenes.split(',')]

    best_frames = {
        1: 98, 2: 164, 3: 166, 4: 460, 5: 88, 6: 395,
        7: 244, 8: 365, 9: 164, 10: 521, 11: 498, 12: 331, 13: 110
    }

    for scene in scene_list:
        print(f"\n{'='*50}")
        print(f"Processing Scene {scene}...")
        print(f"{'='*50}")
        
        # Determine which frame to use for this scene
        current_frame = args.frame
        if current_frame == 98 and scene in best_frames:
            current_frame = best_frames[scene]
            
        # 1. Extract audio for the specific frame if requested or if missing
        scene_dir = f"work_dirs/vggt_experiment_scene_{scene}"
        pred_wav = f"{scene_dir}/{current_frame}_pred.wav"
        gt_wav = f"{scene_dir}/{current_frame}_gt.wav"
        
        if args.extract or not os.path.exists(pred_wav):
            print(f"Extracting specific audio for frame {current_frame}...")
            resume_path = f"logs/vggt_experiment_scene_{scene}/vggt_experiment_scene_{scene}/100.pth"
            if os.path.exists(resume_path):
                subprocess.run([
                    "python", "visualisation/save_frame_audio.py",
                    "--yaml_file", "configs/rwavs.yaml",
                    "--scene", str(scene),
                    "--frame", str(current_frame),
                    "--n_frames", str(args.n_frames),
                    "--resume_path", resume_path
                ])
                # Note: save_frame_audio.py currently saves to work_dirs/representative_results/...
                # We should move it to the scene directory for easier access by plot scripts
                rep_dir = f"work_dirs/representative_results/scene_{scene}_n{args.n_frames}"
                rep_pred = f"{rep_dir}/{current_frame}_pred.wav"
                rep_gt = f"{rep_dir}/{current_frame}_gt.wav"
                if os.path.exists(rep_pred):
                    os.rename(rep_pred, pred_wav)
                    os.rename(rep_gt, gt_wav)
                    print(f"Verified extract saved to {pred_wav}")
            else:
                print(f"Warning: Checkpoint not found at {resume_path}. Skipping extraction.")

        # 2. Run Binaural Plot
        binaural_out = f"visualisation/binaural_analysis_scene_{scene}.jpg"
        print(f"Running plot_binaural_results_v3.py...")
        subprocess.run([
            "python", "visualisation/plot_binaural_results_v3.py",
            "--scene", str(scene),
            "--n_frames", str(args.n_frames),
            "--frame", str(current_frame),
            "--output", binaural_out
        ])
        
        # 3. Run Comparative Plot
        print(f"Running plot_comparative_results_v2.py for frame {current_frame}...")
        subprocess.run([
            "python", "visualisation/plot_comparative_results_v2.py",
            "--scene", str(scene),
            "--frame", str(current_frame),
            "--n_frames", str(args.n_frames)
        ])
        
        # Move the comparative plot to visualisation folder
        comp_out = f"final_analysis_scene_{scene}_f{current_frame}.jpg"
        if os.path.exists(comp_out):
            dest = f"visualisation/{comp_out}"
            os.rename(comp_out, dest)
            print(f"Moved comparative analysis to {dest}")

    print("\nAll tasks completed!")

if __name__ == "__main__":
    main()
