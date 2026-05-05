import argparse
import os
import pickle
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='vggt_experiment_scene', help='Prefix of the output_dir used in training')
    parser.add_argument('--split', type=str, default='val', help='Split to aggregate: val or train')
    return parser.parse_args()

def aggregate(prefix, split):
    env_dict = {
        "office": list(range(1, 6)),
        "house": list(range(6, 9)),
        "apartment": list(range(9, 12)),
        "outdoor": list(range(12, 14))
    }

    print(f"\n==========================================")
    print(f" AGGREGATING RESULTS: {split.upper()} SPLIT")
    print(f"==========================================\n")

    overall_metrics = {}
    
    for env_name, scene_ids in env_dict.items():
        print(f"--- Environment: {env_name} ---")
        env_metrics = {}
        valid_scenes = 0

        for sid in scene_ids:
            # Matches naming in train_vggt.sh: vggt_experiment_scene_${i}
            file_path = f'av_results/{prefix}_{sid}_{split}.pkl'
            
            # Fallback for old naming convention if prefix matches
            if not os.path.exists(file_path):
                old_path = f'av_results/{prefix}_{sid}_22050.pkl'
                if os.path.exists(old_path):
                    file_path = old_path

            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        metrics = pickle.load(f)
                    
                    scene_str = f"  Scene {sid:<2}: "
                    found_keys = False
                    for k, v in metrics.items():
                        if k in ['mag', 'lre', 'dpam', 'rte', 'env']:
                            val = np.array(v).flatten()[0]
                            env_metrics.setdefault(k, []).append(val)
                            overall_metrics.setdefault(k, []).append(val)
                            scene_str += f"{k.upper()}: {val:.4f}  "
                            found_keys = True
                    
                    if found_keys:
                        print(scene_str)
                        valid_scenes += 1
                except Exception as e:
                    print(f"  [ERROR] Scene {sid}: Could not read {file_path} ({e})")
            else:
                pass # Silent skip to avoid clutter if some scenes aren't processed yet

        if valid_scenes > 0:
            for k, vals in env_metrics.items():
                avg = np.mean(vals)
                print(f"  {k.upper():<5}: {avg:.4f}")
        else:
            print("  No data available.")
        print("")

    print("==========================================")
    print(" OVERALL AVERAGE (13 SCENES)")
    print("==========================================")
    if overall_metrics:
        for k, vals in overall_metrics.items():
            avg = np.mean(vals)
            print(f"{k.upper():<5}: {avg:.4f}")
    else:
        print("No overall data found.")
    print("==========================================\n")

if __name__ == "__main__":
    args = parse_args()
    aggregate(args.prefix, args.split)
