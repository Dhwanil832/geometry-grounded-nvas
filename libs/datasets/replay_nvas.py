"""
Visual Geometry Grounded Novel-View Acoustic Synthesis
CVPR 2026 Workshop — Purdue University Northwest

Replay-NVAS dataset loader adapted from:
  AV-Cloud: Spatial Audio Rendering Through Audio-Visual Cloud Splatting
  Mingfei Chen, University of Washington NeuroAI Lab
  https://github.com/yoyomimi/AV-Cloud
"""



import json
import os
import pickle
import random
import warnings

import einops
import julius
import librosa
import numpy as np
import torch
import torchaudio
import torchvision
from decord import AudioReader, VideoReader, cpu
from PIL import Image
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, sosfiltfilt
from torch.utils.data import Dataset

from libs.datasets.scene import Scene
from libs.datasets.scene.gaussian_model import GaussianModel

warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter("ignore", UserWarning)



SCENE_SPLITS = {
    'v3': {
        'train': ['SC-1025',  'SC-1032', 'SC-1033',  'SC-1045', 'SC-1059', 'SC-1069', 'SC-1078',
                  'SC-1066',  'SC-1070', 'SC-1076',  'SC-1081', 'SC-1077', 'SC-1035', 
                  'SC-1085', 'SC-1087',  'SC-1101', 'SC-1102', 'SC-1104', 'SC-1106', 'SC-1105', 'SC-1090',
                  'SC-1036', 'SC-1048', 'SC-1049', 'SC-1075', 'SC-1108', 'SC-1047', 'SC-1050'], # 28
        'val': ['SC-1026', 'SC-1054', 'SC-1073', 'SC-1082', 'SC-1092', 'SC-1103'],  # 6
        'test': ['SC-1027', 'SC-1044', 'SC-1052', 'SC-1074', 'SC-1084', 'SC-1093', 'SC-1107']  # 7

    },
}


def to_tensor(v):
    import numpy as np

    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y


def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16, return_second=True):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''

    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)

    if not return_second:
        tau *= fs

    return tau, cc


def read_clip(img_file, split, target_sr):
    video_file = img_file.replace('.png', '.mp4')
    binaural_file = img_file.replace('.png', '.wav')
    binaural_pkl_file = img_file.replace('.png', f'_{target_sr}.pkl')
    if os.path.exists(img_file):
        rgb = np.array(Image.open(img_file))
    else:
        print('Imag file does not exist: ', img_file)
        vr = VideoReader(video_file, ctx=cpu(0))
        rgb = vr[np.random.randint(0, len(vr)) if split == 'train' else len(vr) // 2].asnumpy()

    if os.path.exists(binaural_pkl_file) and os.path.getsize(binaural_pkl_file) > 0:
        try:
            with open(binaural_pkl_file, 'rb') as file:
                audio = pickle.load(file)
        except (EOFError, pickle.UnpicklingError):
            print(f'[Dataset] Corrupted pickle detected at {binaural_pkl_file}. Regenerating...')
            audio, _ = librosa.load(binaural_file, sr=target_sr, mono=False)
            with open(binaural_pkl_file, 'wb') as file:
                pickle.dump(audio, file)
    else:
        audio, input_sr = librosa.load(binaural_file, sr=target_sr, mono=False)
        with open(binaural_pkl_file, 'wb') as file:
            pickle.dump(audio, file)


    return rgb, audio


def read_near_audio(binaural_file, target_sr):
    binaural_pkl_file = binaural_file.replace('.wav', f'_{target_sr}.pkl')
    if os.path.exists(binaural_pkl_file) and os.path.getsize(binaural_pkl_file) > 0:
        try:
            with open(binaural_pkl_file, 'rb') as file:
                audio = pickle.load(file)
        except (EOFError, pickle.UnpicklingError):
            print(f'[Dataset] Corrupted near-audio pickle at {binaural_pkl_file}. Regenerating...')
            audio, _ = librosa.load(binaural_file, sr=target_sr, mono=False)
            with open(binaural_pkl_file, 'wb') as file:
                pickle.dump(audio, file)
    else:
        audio, input_sr = librosa.load(binaural_file, sr=target_sr, mono=False)
        with open(binaural_pkl_file, 'wb') as file:
            pickle.dump(audio, file)
    audio = audio.reshape(-1)
    audio = np.stack([audio, audio], axis=0)
    return audio


    
class ReplayNVASDataset(Dataset):
    def __init__(self, 
                 cfg,
                 data_root,
                 split='train',
                 scene=None,
                 gaussian=None
                 ):
        super().__init__()
        self.split = split
        self.cfg = cfg
        self.estimated_heads = None
        self.audio_len = self.sr = self.cfg.dataset.sr

        with open(os.path.join(data_root, 'v3/metadata_v2.json'), 'r') as fo:
            self.metadata = json.load(fo)
        scene_splits = SCENE_SPLITS['v3']
        ori_clip_dirs = [x.replace('data/appen_dataset', data_root) for x in list(self.metadata.keys()) if x.split('/')[-2] in scene_splits[split]]
        
        train_frames_end = int(len(ori_clip_dirs) * 0.8)

        self.gaussians = None
        self.scene = type('Scene', (), {'mid_init': None})()

        # ------------------------------------------------------------------ #
        # Context Selection and VGGT Features
        # ------------------------------------------------------------------ #
        n_ctx = getattr(cfg.dataset, 'N_points', 256)
        selection_path = os.path.join(data_root, f"replaynavs_outputs_{n_ctx}/context_selection.json")
        features_path = os.path.join(data_root, f"replaynavs_outputs_{n_ctx}/context_features.pth")
        
        if os.path.exists(selection_path):
            with open(selection_path, 'r') as f:
                selection_data = json.load(f)
            self.context_poses = torch.tensor([f['pose_enc'] for f in selection_data['frames']], dtype=torch.float32)
        else:
            print(f"[Dataset] Warning: selection not found at {selection_path}")
            self.context_poses = None

        if os.path.exists(features_path):
            # context_features.pth is likely (256, 2048) or similar
            self.context_features = torch.load(features_path, map_location='cpu')
            if isinstance(self.context_features, dict) and 'features' in self.context_features:
                self.context_features = self.context_features['features']
            if self.context_features.dim() == 3: # (256, 5, 2048)
                self.context_features = self.context_features[:, 0, :] # Get only the camera token (index 0)
        else:
            print(f"[Dataset] Warning: features not found at {features_path}")
            self.context_features = None

        # ------------------------------------------------------------------ #
        # Learnable M-Value Initialization (V-replacement)
        # Load precomputed M-cache and compute STFT magnitude fingerprint
        # ------------------------------------------------------------------ #
        self.scene.mid_init = None
        if split == 'train':
            mid_cache_path = os.path.join(data_root, f"replaynavs_outputs_{n_ctx}/mid_signal_cache.json")
            
            if os.path.exists(mid_cache_path) and self.context_poses is not None:
                print(f"[Dataset] Loading M-cache from {mid_cache_path}")
                with open(mid_cache_path, 'r') as f:
                    mid_cache = json.load(f)
                
                # Same STFT parameters as used in GGAD forward pass
                stft_n_fft = 512
                stft_hop   = 127
                stft_win   = 512
                hann_win   = torch.hann_window(stft_win)
                
                mags = []
                # selection_data['frames'] contains the ordered list of 256 context frames
                for frame_info in selection_data['frames']:
                    sid = frame_info['scene_id']
                    iid = frame_info['image_id']
                    if sid in mid_cache and iid in mid_cache[sid]:
                        m_wav = torch.tensor(mid_cache[sid][iid], dtype=torch.float32)
                        spec  = torch.stft(m_wav, n_fft=stft_n_fft, hop_length=stft_hop,
                                           win_length=stft_win, window=hann_win,
                                           center=True, return_complex=True)
                        mag = spec.abs().mean(dim=-1) # (freq_num,)
                        mags.append(mag)
                
                if len(mags) == 256:
                    self.scene.mid_init = torch.stack(mags) # (256, freq_num)
                    print(f"[Dataset] Initialized mid_init with shape {self.scene.mid_init.shape}")
                else:
                    print(f"[Dataset] Warning: Found {len(mags)} mags, expected 256. Skipping mid_init.")
            else:
                print(f"[Dataset] Warning: M-cache or selection not found at {mid_cache_path}")
        
        # Share the same scene object with mid_init between train/val if provided
        if scene is not None and hasattr(scene, 'mid_init'):
            self.scene.mid_init = scene.mid_init


        self.train_cams = [1, 2, 3, 4, 5, 6, 7, 8]
        self.test_cams = [1, 2, 3, 4, 5, 6, 7, 8]

        self.pair_list = []
        self.clip_dirs = []
        self.cam_dict = {}

        # Load VGGT predicted poses (9-float vectors: [tx, ty, tz, qw, qx, qy, qz, fov_h, fov_w])
        pose_json_path = os.path.join(data_root, 'replaynavs_outputs_256/target_poses.json')
        with open(pose_json_path, 'r') as file:
            self.cam_dict = json.load(file)
            
        for clip_idx, clip_dir in enumerate(ori_clip_dirs):
            # For ReplayNVAS, we can use most clips from the metadata
            # (Filtering logic can be adjusted if needed, but keeping modulo 3 for now)
            if clip_idx % 3 != 0:
                continue
            self.clip_dirs.append(clip_dir)


        if split == 'train':
            for clip_dir in self.clip_dirs:
                # clip_dir is like 'data/appen_dataset/v3/SC-1025/0'
                scene_id = clip_dir.split('/')[-2]
                for cam_id in range(1, 9):
                    file = os.path.join(clip_dir, f'{cam_id}.png')
                    key = f"{scene_id}_{cam_id}"
                    if os.path.exists(file) and key in self.cam_dict:
                        self.pair_list.append((clip_dir, file))
        elif split == 'val':
            for clip_dir in self.clip_dirs:
                scene_id = clip_dir.split('/')[-2]
                for cam_id in range(1, 9):
                    file = os.path.join(clip_dir, f'{cam_id}.png')
                    key = f"{scene_id}_{cam_id}"
                    if os.path.exists(file) and key in self.cam_dict:
                        self.pair_list.append((clip_dir, file))
        elif split == 'test':
            for clip_dir in self.clip_dirs:
                scene_id = clip_dir.split('/')[-2]
                for cam_id in range(1, 9):
                    file = os.path.join(clip_dir, f'{cam_id}.png')
                    key = f"{scene_id}_{cam_id}"
                    if os.path.exists(file) and key in self.cam_dict:
                        self.pair_list.append((clip_dir, file))
        
        print(f'Number of clip is {len(self.clip_dirs)} for {self.split.upper()}')
        print(f'Number of samples is {len(self.pair_list)} for {self.split.upper()}')

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, index):
        clip_dir, file = self.pair_list[index]
        file_1, file_2 = file, file
        
        frame_idx = int(file_1.split('/')[-2])
        v2, a2 = read_clip(file_2, self.split, self.sr)
        # source
        near_file = os.path.join(clip_dir, 'near.wav')
        source_gt_audio = read_near_audio(near_file, self.sr)
        a1 = source_gt_audio
        input_tgt_rgb = to_tensor(v2).permute(2, 0, 1) / 255.0
        # True in the ori paper, remove low band noise
        a1 = np.array(butter_bandpass_filter(a1, 150, self.sr // 2 - 1, self.sr, order=5).copy(), dtype=np.float32)
        a2 = np.array(butter_bandpass_filter(a2, 150, self.sr // 2 - 1, self.sr, order=5).copy(), dtype=np.float32)

        # True in the ori paper, remove delay
        delay = int(gcc_phat(np.mean(a2, axis=0), np.mean(a1, axis=0), self.sr,
                                    return_second=False)[0])
        delay = np.clip(delay, a_min=-200, a_max=200)
        if delay > 0:
            a1 = np.pad(a1, ((0, 0), (delay, 0)))[:, :-delay]
        else:
            a1 = np.pad(a1, ((0, 0), (0, -delay)))[:, -delay:]


        sample = dict()
        a1, a2 = self.process_audio(a1), self.process_audio(a2)

        # Smart Query consistency: Always take the first 1 second (deterministic)
        input_src_wav = to_tensor(a1[:, :self.audio_len])
        gt_audio = torch.from_numpy(a2[:, :self.audio_len])

        tgt_index = int(file_2.split('/')[-1][:-4])
        scene_id = file_2.split('/')[-3]
        key = f"{scene_id}_{tgt_index}"
        
        # Poses are 9-D: [tx, ty, tz, qw, qx, qy, qz, fov_h, fov_w]
        tgt_pose_9d = np.array(self.cam_dict[key], dtype=np.float32)
        cam_pose = torch.from_numpy(tgt_pose_9d)

        # RETURN: index, target_pose, context_vggt_tokens, context_poses, gt_audio, source_gt_audio
        if self.context_features is None:
            raise RuntimeError(
                f"[Dataset] context_features not loaded — run feature extraction first.\n"
                f"See: feature_extraction/replaynavs/extract_context.py"
            )
        if self.context_poses is None:
            raise RuntimeError(
                f"[Dataset] context_poses not loaded — selection.json missing or invalid.\n"
                f"See: feature_extraction/replaynavs/extract_targets.py"
            )
        return index, cam_pose.float(), self.context_features.float(), self.context_poses.float(), gt_audio.float(), input_src_wav.float()

            
    def process_audio(self, audio):
        if audio.shape[1] < self.audio_len:
            audio = np.pad(audio, ((0, 0), (0, self.audio_len - audio.shape[1])))
        return audio


def build_dataset(cfg):
    train_dataset = ReplayNVASDataset(cfg, cfg.dataset.data_root, split='train')
    eval_dataset = ReplayNVASDataset(cfg, cfg.dataset.data_root, split='val', scene=train_dataset.scene, gaussian=train_dataset.gaussians)
    return train_dataset, eval_dataset

def build_single_dataset(cfg, split):
    single_dataset = ReplayNVASDataset(cfg, cfg.dataset.data_root, split=split)
    return single_dataset
