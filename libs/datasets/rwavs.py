"""
Visual Geometry Grounded Novel-View Acoustic Synthesis
CVPR 2026 Workshop — Purdue University Northwest

RWAVS dataset loader extended from:
  AV-Cloud: Spatial Audio Rendering Through Audio-Visual Cloud Splatting
  Mingfei Chen, University of Washington NeuroAI Lab
  https://github.com/yoyomimi/AV-Cloud
"""



import json
import os
import pickle
import sys

import imageio
import librosa
import numpy as np
import torch
import torchaudio
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from configs import cfg
from libs.datasets.scene import Scene
from libs.datasets.scene.gaussian_model import GaussianModel


class SceneDataset(torch.utils.data.Dataset):
    def __init__(self,
                 cfg,
                 data_root,
                 split='train', 
                 scene=None,
                 gaussian=None
                 ):
        super(SceneDataset, self).__init__()
        self.data_root = data_root    
        self.cfg = cfg    
        video_name = cfg.dataset.video.split('_')[-1]
        
        # PURE GEOMETRY: No COLMAP, No SFM, No Gaussian Splatting initialization.
        self.gaussians = None
        # Lightweight scene container for model initialization data (M-cache)
        self.scene = type('Scene', (), {'mid_init': None})()
        # OLD LOGIC REMOVED: No more gs_cameras.json or transforms_scale_{split}.json
        # NEW LOGIC: Load from selection.json (Single Source of Truth)
        # selection.json contains: Scene IDs, and for each scene, the 256 Context Poses and feature path.
        
        # Path: RWAVS/vggt_outputs_v3/selection.json
        selection_path = f'{cfg.dataset.data_root.replace("release", "vggt_outputs_v3")}/selection.json'
        
        # We also need original transforms.json for the target pose
        transforms_path = f'{cfg.dataset.data_root}/{video_name}/transforms_{split}.json'
        
        if not os.path.exists(selection_path):
             raise FileNotFoundError(f"Selection file not found: {selection_path}")
        if not os.path.exists(transforms_path):
             raise FileNotFoundError(f"Transforms file not found: {transforms_path}")

        with open(selection_path, 'r') as file:
            selection_data = json.load(file)
        with open(transforms_path, 'r') as file:
            transforms_data = json.load(file)
            
        # Map frame ID to transform matrix
        gt_transforms = {}
        for frame in transforms_data['frames']:
            # file_path is usually "frames/00001" or "frames/00001.png"
            base_name = os.path.basename(frame['file_path'])
            # Remove any extension (.png, .jpg, etc)
            clean_id = os.path.splitext(base_name)[0]
            # Zero-pad to 5 digits to match selection.json (and target extraction)
            clean_id = f"{int(clean_id):05d}"
            gt_transforms[clean_id] = frame['transform_matrix']

        self.cam_dict = {} # Stores (target_pose, context_poses, feature_path)
        self.path_list = []
        
        # Coordinate flip matrix: Colmap (OpenCV) -> NeRF (OpenGL) backwards?
        flip_mat = np.diag([1, -1, -1])

        # 1. Context Poses & Features (Same for all targets in this scene)
        if video_name not in selection_data:
             raise ValueError(f"Scene {video_name} is missing from {selection_path}")
             
        scene_data = selection_data[video_name]
        
        # Extract the 256 Context Poses
        context_poses = []
        for item in scene_data['frames']:
            c_pose = np.array(item['pose_enc'], dtype=np.float32) # 9D
            context_poses.append(c_pose)
        context_poses = np.stack(context_poses) # (256, 9)
        
        # Feature File Path
        # selection.json has relative path: "1/features.pth"
        # We need absolute: root/vggt_outputs_v3/1/features.pth
        feat_rel_path = scene_data['feature_file']
        feat_abs_path = os.path.join(os.path.dirname(selection_path), feat_rel_path)

        # 2. Iterate over valid targets for this split
        for frame_id in gt_transforms.keys():
            
            # Target Pose (From GT)
            tgt_matrix_list = gt_transforms[frame_id]
            t_matrix = np.array(tgt_matrix_list)
            t_center = t_matrix[:3, 3]
            t_rot = t_matrix[:3, :3] @ flip_mat
            target_pose = np.hstack([t_center, t_rot.reshape(-1)]) # 12D
            
            # Store in dictionary
            # Key: frame_id (e.g., "00001")
            self.cam_dict[frame_id] = {
                'target_pose': target_pose,
                'context_poses': context_poses,
                'feature_path': feat_abs_path
            }
            
            # Append to path list for indexing
            self.path_list.append((video_name, frame_id))

        self.split = split

        audio_len = self.cfg.dataset.sr
        self.sr = audio_len
 
        audio_list_path = f'{data_root}/{video_name}/22050_audio_{split}_48k.pkl'
        if not os.path.exists(audio_list_path):
            # Fallback to creating pickle (assumes wav files exist)
            # Note: keeping original logic, but simplified path handling might be needed if paths changed
            audio_path = os.path.join(self.data_root, video_name, 'binaural_syn_re.wav')
            source_audio_path = os.path.join(self.data_root, video_name, 'source_syn_re.wav')
            
            # Careful: data_root might point to release/3, but we need release_output for features.
            # Audio is likely still in release/3.
            
            full_gt_audio, audio_rate = librosa.load(audio_path , sr=self.sr, mono=False)
            source_full_gt_audio, audio_rate = librosa.load(source_audio_path , sr=self.sr, mono=False)
            self.audio_list = {}
            self.audio_list[video_name] = {}
            self.audio_list[video_name]['gt'] = full_gt_audio
            self.audio_list[video_name]['source'] = source_full_gt_audio
            with open(audio_list_path, 'wb') as file:
                pickle.dump(self.audio_list, file)
        else:
            with open(audio_list_path, 'rb') as file:
                self.audio_list = pickle.load(file)


        print(f'{split} dataset len: {len(self.path_list)}')

        # ------------------------------------------------------------------ #
        # Learnable M-Value Initialization (V-replacement)
        # Load precomputed M-cache and compute STFT magnitude fingerprint
        # ------------------------------------------------------------------ #
        self.mid_init = None
        if split == 'train':
            mid_cache_path = selection_path.replace("selection.json", "mid_signal_cache.json")
            if os.path.exists(mid_cache_path):
                print(f"[Dataset] Loading M-cache from {mid_cache_path}")
                with open(mid_cache_path, 'r') as f:
                    mid_cache = json.load(f)
                
                scene_cache = mid_cache.get(video_name, {})
                if scene_cache:
                    # Same STFT parameters as used in GGAD forward pass
                    stft_n_fft = 512
                    stft_hop   = 127
                    stft_win   = 512
                    hann_win   = torch.hann_window(stft_win)
                    
                    mags = []
                    # selection_data[video_name]['frames'] contains the ordered list of context frames
                    for frame_info in scene_data['frames']:
                        fid = frame_info['frame_id']
                        if fid in scene_cache:
                            m_wav = torch.tensor(scene_cache[fid], dtype=torch.float32)
                            spec  = torch.stft(m_wav, n_fft=stft_n_fft, hop_length=stft_hop,
                                               win_length=stft_win, window=hann_win,
                                               center=True, return_complex=True)
                            mag = spec.abs().mean(dim=-1) # (freq_num,)
                            mags.append(mag)
                    
                    if mags:
                        self.scene.mid_init = torch.stack(mags) # (256, freq_num)
                        print(f"[Dataset] Initialized mid_init with shape {self.scene.mid_init.shape}")
            else:
                print(f"[Dataset] Warning: M-cache not found at {mid_cache_path}")
        
        # Share the same scene object with mid_init between train/val
        self.mid_init = self.scene.mid_init



    def __getitem__(self, index):
        video_name, frame_id = self.path_list[index]
        audio_len = self.sr
        
        # Retrieve pre-processed data
        entry = self.cam_dict[frame_id]
        
        target_pose = torch.from_numpy(entry['target_pose']).float() # (12,)
        context_poses = torch.from_numpy(entry['context_poses']).float() # (32, 12)
        
        # Load Context VGGT Camera/Patch Features (Strictly spatial, no register tokens here)
        # Shape: (32, 5, 2048) -> (32, 2048)
        vggt_all_tokens = torch.load(entry['feature_path'], map_location='cpu')
        vggt_tokens = vggt_all_tokens[:, 0, :] # Extract only the camera token (index 0)
        
        # Audio Processing (using integer index from frame_id)
        start_idx = int(frame_id)
        ori_start_idx = start_idx
        
        gt_audio = torch.zeros(2, audio_len)
        cur_gt_audio = torch.from_numpy(self.audio_list[video_name]['gt'][..., (start_idx-1)*audio_len:start_idx*audio_len])
        gt_audio[..., :cur_gt_audio.shape[-1]] = cur_gt_audio.float()

        source_gt_audio = torch.zeros(2, audio_len)
        source_cur_gt_audio = torch.from_numpy(self.audio_list[video_name]['source'][..., (start_idx-1)*audio_len:start_idx*audio_len])
        source_gt_audio[..., :source_cur_gt_audio.shape[-1]] = source_cur_gt_audio.float()
       
        # RETURN: target_pose, vggt_tokens, context_poses, audio...
        return ori_start_idx, target_pose, vggt_tokens, context_poses, gt_audio, source_gt_audio

    def __len__(self):
        return len(self.path_list)



def build_dataset(cfg):
    train_dataset = SceneDataset(cfg, cfg.dataset.data_root, split='train')
    eval_dataset = SceneDataset(cfg, cfg.dataset.data_root, split='val', scene=train_dataset.scene, gaussian=train_dataset.gaussians)
    return train_dataset, eval_dataset

def build_single_dataset(cfg, split):
    single_dataset = SceneDataset(cfg, cfg.dataset.data_root, split=split)
    return single_dataset
