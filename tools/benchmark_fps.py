import sys
import os
import torch
import time
import os.path as osp
import argparse
from tqdm import tqdm

project_path = osp.abspath(osp.curdir)
sys.path.insert(0, project_path)

from configs import cfg, update_config
from importlib import import_module as impm

def benchmark():
    # Mock args for update_config
    args = argparse.Namespace()
    args.yaml_file = 'configs/replaynvas.yaml'
    args.opts = []
    
    update_config(cfg, args)
    
    # Use build_single_dataset to get validation set
    dataset = getattr(impm(f'libs.datasets.{cfg.dataset.name}'), 'build_single_dataset')(cfg, 'val')
    model_file = cfg.model.file
    
    model = getattr(impm(f'libs.models.{model_file}'), 'build_model')(cfg, dataset.scene).cuda()
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Real data inputs for FPS benchmark - Entire Val Set
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.workers
    )
    
    samples = []
    print(f"Pre-loading {len(dataset)} samples from {cfg.dataset.name.upper()} Val Set...")
    for data in tqdm(test_loader, desc="Pre-loading samples"):
        # data format from our updated replay_nvas.py: 
        # index, target_pose, context_vggt_tokens, context_poses, gt_audio, source_gt_audio
        _, target_pose, vggt_tokens, context_poses, gt_audio, source_gt_audio = data
        samples.append({
            'target_pose': target_pose.cuda(),
            'vggt_tokens': vggt_tokens.cuda(),
            'context_poses': context_poses.cuda(),
            'source_gt_audio': source_gt_audio.cuda()
        })
    
    model.eval()
    with torch.no_grad():
        # Warmup
        print("Warming up (10 iterations)...")
        for i in range(10):
            s = samples[i % len(samples)]
            _ = model(target_pose=s['target_pose'], 
                      vggt_tokens=s['vggt_tokens'], 
                      context_poses=s['context_poses'], 
                      source_gt_audio=s['source_gt_audio'], 
                      is_val=True)
        
        torch.cuda.synchronize()
        print(f"Benchmarking on {len(samples)} samples...")
        start = time.time()
        for s in samples:
            _ = model(target_pose=s['target_pose'], 
                      vggt_tokens=s['vggt_tokens'], 
                      context_poses=s['context_poses'], 
                      source_gt_audio=s['source_gt_audio'], 
                      is_val=True)
        torch.cuda.synchronize()
        end = time.time()
        
    avg_time = (end - start) / len(samples)
    fps = 1.0 / avg_time
    
    print("-" * 30)
    print(f"# Model: {model_file}")
    print(f"# Dataset: {cfg.dataset.name}")
    print(f"# Params: {n_params / 1e6:.2f}M")
    print(f"# Real Data FPS (on {len(samples)} frames): {fps:.2f}")
    print("-" * 30)

if __name__ == '__main__':
    benchmark()
