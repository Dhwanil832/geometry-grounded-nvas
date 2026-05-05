import sys
import os
import os.path as osp
import argparse
import torch
import numpy as np
import soundfile as sf
from importlib import import_module as impm

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(osp.abspath(__file__))
project_path = osp.dirname(this_dir)
add_path(project_path)
for folder in ['criterions', 'datasets', 'evaluators', 'models', 'renders', 'trainers', 'utils']:
    add_path(osp.join(project_path, 'libs', folder))

from configs import cfg, update_config
from libs.utils import misc

def main():
    parser = argparse.ArgumentParser(description='Save audio for a specific frame')
    parser.add_argument('--yaml_file', required=True, type=str, help='experiment configure file name')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n_frames', type=int, default=128)
    parser.add_argument('--scene', type=int, default=1)
    parser.add_argument('--frame', type=int, default=349)
    parser.add_argument('--resume_path', required=True, type=str)
    parser.add_argument('--opts', help="Modify config options", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Update config
    update_config(cfg, args)
    cfg.defrost()
    cfg.gpu = args.gpu
    cfg.dataset.n_frames = args.n_frames
    cfg.device = f'cuda:{args.gpu}'
    cfg.dataset.video = f'_{args.scene}'
    cfg.freeze()

    device = torch.device(cfg.device)
    torch.cuda.set_device(args.gpu)

    # Dataset
    print(f"Loading dataset for scene {args.scene}...")
    dataset = getattr(impm(cfg.dataset.name), 'build_single_dataset')(cfg, 'val')
    
    # Find the data for the specific frame
    target_frame_id = f"{args.frame:05d}"
    found_idx = -1
    for i, (vname, fid) in enumerate(dataset.path_list):
        if fid == target_frame_id:
            found_idx = i
            break
            
    if found_idx == -1:
        print(f"Could not find frame {target_frame_id} in val dataset. Trying train dataset...")
        dataset = getattr(impm(cfg.dataset.name), 'build_single_dataset')(cfg, 'train')
        for i, (vname, fid) in enumerate(dataset.path_list):
            if fid == target_frame_id:
                found_idx = i
                break

    if found_idx == -1:
        print(f"Could not find frame {target_frame_id} in train dataset either.")
        return

    # Model
    print(f"Building model {cfg.model.file}...")
    model = getattr(impm(cfg.model.file), 'build_model')(cfg, dataset.scene).to(device)
    
    print(f"Loading checkpoint from {args.resume_path}...")
    checkpoint = torch.load(args.resume_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    # Handle DataParallel prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()

    # Get data
    print(f"Extracting data for frame {args.frame}...")
    batch = dataset[found_idx]
    # batch: ori_start_idx, target_pose, vggt_tokens, context_poses, gt_audio, source_gt_audio
    _, target_pose, vggt_tokens, context_poses, gt_audio, source_gt_audio = batch
    
    # To device and add batch dim
    target_pose = target_pose.to(device).unsqueeze(0)
    vggt_tokens = vggt_tokens.to(device).unsqueeze(0)
    context_poses = context_poses.to(device).unsqueeze(0)
    source_gt_audio = source_gt_audio.to(device).unsqueeze(0)

    # Inference
    print("Running inference...")
    with torch.no_grad():
        pred_wav = model(target_pose=target_pose, vggt_tokens=vggt_tokens, context_poses=context_poses, source_gt_audio=source_gt_audio, is_val=True)
    
    # Save
    out_dir = f"work_dirs/representative_results/scene_{args.scene}_n{args.n_frames}"
    os.makedirs(out_dir, exist_ok=True)
    
    # Shape of pred_wav is likely (1, 2, L) or (2, L)
    pred_audio = pred_wav.cpu().numpy().squeeze()
    if pred_audio.ndim == 2:
        pred_audio = pred_audio.T # (L, 2)
        
    gt_audio_np = gt_audio.cpu().numpy().squeeze().T
    
    out_pred = f"{out_dir}/{args.frame}_pred.wav"
    out_gt = f"{out_dir}/{args.frame}_gt.wav"
    
    sf.write(out_pred, pred_audio, cfg.dataset.sr)
    sf.write(out_gt, gt_audio_np, cfg.dataset.sr)
    print(f"Successfully saved results to:")
    print(f"  Pred: {out_pred}")
    print(f"  GT:   {out_gt}")

if __name__ == "__main__":
    main()
