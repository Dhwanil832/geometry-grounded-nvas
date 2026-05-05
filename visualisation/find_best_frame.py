import sys
import os.path as osp
import torch
import numpy as np
import argparse
from importlib import import_module as impm

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

project_path = osp.abspath(osp.curdir)
sys.path.insert(0, project_path)
for folder in ['criterions', 'datasets', 'evaluators', 'models', 'renders', 'trainers', 'utils']:
    add_path(osp.join(project_path, 'libs', folder))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=int, default=1)
    args = parser.parse_args()

    from configs import cfg
    cfg.defrost()
    cfg.dataset.data_root = 'RWAVS/release'
    cfg.dataset.n_frames = 128
    cfg.dataset.video = f'_{args.scene}'
    cfg.freeze()

    dataset = getattr(impm('libs.datasets.rwavs'), 'build_single_dataset')(cfg, 'val')
    best_e = -1
    best_f = None
    
    print(f"Dataset size: {len(dataset)}")
    for i in range(len(dataset)):
        batch = dataset[i]
        fid = dataset.path_list[i][1]
        gt_audio = batch[4]
        e = torch.sum(gt_audio**2).item()
        if e > best_e:
            best_e = e
            best_f = fid

    print(f"Best val frame: {best_f}, Energy: {best_e}")
