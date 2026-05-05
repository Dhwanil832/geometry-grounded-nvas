"""
Code of "Visual Geometry Grounded Novel-View Acoustic Synthesis"

CVPR 2026 Workshop
Purdue University Northwest
"""


import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from libs.models.networks.encoder import (PositionalEncoder,
                                          embedding_module_log)
from libs.models.networks.mlp import basic_project2
from libs.models.networks.transformer import *


class GGAD(nn.Module):
    """Geometry-Grounded Acoustic Decoder (GGAD)

    Unified model for Novel-View Acoustic Synthesis on both RWAVS and
    Replay-NVAS datasets. Dataset-specific behaviour is controlled entirely
    through config (pose_dim, sr).
    """

    def __init__(self, cfg, scene, intermediate_ch = 256, time_num = 174, freq_num = 257):

        super(GGAD, self).__init__()
        self.joint_emb_dim = cfg.model.joint_emb_dim
        self.scene = scene
        self.freq_num = freq_num

        # Dynamically compute time frames from dataset sample rate
        # RWAVS  (sr=22050): (22050 // 127) + 1 = 174
        # ReplayNVAS (sr=16000): (16000 // 127) + 1 = 126
        sr = getattr(cfg.dataset, 'sr', 22050)
        self.time_num = (sr // 127) + 1

        self.d_model = intermediate_ch

        # Pose input dimension: 12 for RWAVS, 9 for Replay-NVAS (VGGT format)
        pose_dim = getattr(cfg.model, 'pose_dim', 12)

        # Projectors for VGGT visual tokens and camera poses
        self.vggt_projector = nn.Linear(2048, self.d_model)
        self.pose_projector = nn.Sequential(
            nn.Linear(pose_dim, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )
        self.ctx_pose_projector = nn.Sequential(
            nn.Linear(9, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )
        
        self.diff_prj = nn.Linear(self.d_model, 1)
        self.mix_prj = nn.Linear(self.d_model, 1)
        
        # Learnable positional embedding for Frequency Tokens (Query)
        self.freq_pos_embed = nn.Parameter(torch.randn(self.freq_num, self.d_model) * 0.02)

        # Transformer Decoder setup
        normalize_before = False
        decoder_layer = TransformerDecoderLayer(self.d_model, nhead=4, dim_feedforward=512,
                                                dropout=0.1, activation="relu", normalize_before=normalize_before)
        decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(decoder_layer, 3, decoder_norm, return_intermediate=False)

        # Frequency Embedder (Req #2)
        self.freq_embedder = embedding_module_log(num_freqs=10, ch_dim=1)
        self.query_prj = nn.Sequential(nn.Linear(21, self.d_model), nn.ReLU(inplace=True))

        # Audio Refinement Head (SARH)
        self.out = nn.Sequential(nn.Conv2d(4, 16, 7, 1, 3),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(16, 16, 3, 1, 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(16, 1, 3, 1, 1))

        # ------------------------------------------------------------------ #
        # Learnable M-Value for Cross-Attention
        # Initialize from precomputed M-init provided by Dataset/Scene
        # ------------------------------------------------------------------ #
        n_ctx = 256
        if hasattr(scene, 'mid_init') and scene.mid_init is not None:
            print(f"[Model] Initializing mid_param from scene.mid_init")
            mid_init = scene.mid_init
        else:
            print("[Model] Warning: scene.mid_init not found. Initializing randomly.")
            mid_init = torch.randn(n_ctx, freq_num) * 0.02
            
        self.mid_param = nn.Parameter(mid_init)
        self.mid_projector = nn.Linear(freq_num, self.d_model)


    def forward(self, target_pose, vggt_tokens, context_poses, source_gt_audio, is_val=False):
        """
        Args:
            target_pose: (B, 9). Camera poses of LISTENER (VGGT format).
            vggt_tokens: (B, 256, 2048). Visual features from context.
            context_poses: (B, 256, 9). Camera poses of CONTEXT.
            source_gt_audio: (B, 2, T). Source audio input waveform.
        Return:
            out_pred_wav (B, 2, T). Predict binaural audio at the target viewpoint.
        """
        device = target_pose.device
        B = target_pose.shape[0]
        hop_length = 127
        n_fft = 512
        window_length = 512
        torch_window = torch.hann_window(window_length=window_length).to(device)
        ref_spec_complex = torch.stft(source_gt_audio.detach().mean(1), n_fft=n_fft, hop_length=hop_length, win_length=window_length, center=True, window=torch_window, return_complex=True)
        ref_mag = torch.abs(ref_spec_complex)
        left_phase = right_phase = torch.angle(ref_spec_complex)
        
        # --- 1. Query Construction (Listener) ---
        # Frequency Embeddings (F Tokens)
        freq = torch.linspace(-0.99, 0.99, self.freq_num, device=device).unsqueeze(1)
        freq_enc = self.freq_embedder(freq) # (F, 21)
        
        # PURE GEOMETRY QUERY CONTENT: Freq Embed only 
        query_embed = self.query_prj(freq_enc).unsqueeze(1).repeat(1, B, 1) # (F, B, D)
        
        # We use only the raw audio frequencies (tgt) as the query content
        query_content = query_embed

        # --- 2. Positional Embeddings vs Content (Separation) ---
        # Target Positional Embedding
        target_pose_embed = self.pose_projector(target_pose).unsqueeze(0) # (1, B, D)
        
        # Context Positional Embedding
        ctx_pose_embed = self.ctx_pose_projector(context_poses).permute(1, 0, 2) # (256, B, D)

        # Pure Content Memory (No coordinate pollution)
        vggt_embed = self.vggt_projector(vggt_tokens).permute(1, 0, 2) # (256, B, D)
        memory_content = vggt_embed 

        # --- 3. Transformer Decoding ---
        # query_content goes to tgt.
        # memory_content goes to memory (K).
        # Combine Frequency pos and Target Pose for full query position
        full_query_pos = self.freq_pos_embed.unsqueeze(1).repeat(1, B, 1) + target_pose_embed

        # Build M-Value: project learnable mid_param → (256, B, D)
        mid_value = self.mid_projector(self.mid_param)             # (256, D)
        mid_value = mid_value.unsqueeze(1).expand(-1, B, -1)       # (256, B, D)
        
        hs, attn_weights = self.decoder(
            tgt=query_content, 
            memory=memory_content, 
            pos=ctx_pose_embed,
            query_pos=full_query_pos,
            memory_value=mid_value   # V = learned M embedding; K = VGGT + camera pose
        )
        
        # Removed use_audio_emb: derive mix_feat directly from decoder output
        mix_feat = hs.permute(1, 0, 2) # (B, F, D)
        hs = hs.reshape(1, self.freq_num, B, -1).permute(2, 0, 1, 3) # B, 1, freq_num, C

        mag_mask = self.mix_prj(mix_feat).expand(B, self.freq_num, self.time_num)
        diff = 2*self.diff_prj(hs).sigmoid() - 1
        diff = diff.reshape(B, self.freq_num, 1).expand(B, self.freq_num, self.time_num)

        # --- 4. Spatial Audio Render Head (SARH) ---
        # Base mono reconstruction
        reconstr_mono = ref_mag * mag_mask
        reconstr_diff = reconstr_mono * diff
        
        left_base = reconstr_mono - reconstr_diff
        right_base = reconstr_mono + reconstr_diff
        
        # Audio Refinement (residual)
        left_in = torch.stack([ref_mag, mag_mask, -diff, left_base], 1)
        right_in = torch.stack([ref_mag, mag_mask, diff, right_base], 1)
        
        left_mag_res = self.out(left_in).squeeze(1)
        right_mag_res = self.out(right_in).squeeze(1)
        
        # Final reconstruction with polar coordinates
        reconstructed_stft_left = torch.polar(F.relu(left_base + left_mag_res), left_phase)
        reconstructed_stft_right = torch.polar(F.relu(right_base + right_mag_res), right_phase)
        reconstructed_stft = torch.stack([reconstructed_stft_left, reconstructed_stft_right]).permute(1, 0, 2, 3).flatten(0, 1)
        
        out_pred_wav = torch.istft(reconstructed_stft, n_fft=n_fft, hop_length=hop_length, win_length=window_length, window=torch_window, center=True).reshape(B, 2, -1)
        
        return out_pred_wav



def build_model(cfg, scene):
    model = GGAD(cfg, scene)
    return model