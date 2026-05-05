import os
import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from PIL import Image

def draw_method_subplot(fig, gs_pos, audio, sr, title, color_lr=['#00e5ff', '#ff00ff']):
    """Draws spectrogram + L/R small waveforms in a clean column."""
    # Nested GridSpec for this slot
    inner_gs = gs_pos.subgridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
    
    # 1. Spectrogram
    ax_spec = fig.add_subplot(inner_gs[0])
    # Use L channel for spectrogram (representative)
    spec = librosa.stft(audio[:, 0], n_fft=512, hop_length=128)
    db = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
    librosa.display.specshow(db, sr=sr, hop_length=128, ax=ax_spec, cmap='viridis', vmin=-60, vmax=0)
    ax_spec.set_title(title, fontsize=16, fontweight='bold', pad=12, color='black')
    ax_spec.set_xticks([])
    ax_spec.set_yticks([])

    # 2. Waveforms (L and R side-by-side)
    inner_wav_gs = inner_gs[1].subgridspec(1, 2, wspace=0.1)
    for i in range(2):
        ax_wav = fig.add_subplot(inner_wav_gs[i])
        ax_wav.plot(audio[:, i], color=color_lr[i], linewidth=0.8)
        # Dynamic scaling for better visibility
        limit = max(0.1, np.max(np.abs(audio)) * 1.1)
        ax_wav.set_ylim(-limit, limit)
        ax_wav.axis('off')
    
    return True

def plot_final_comparison(scene_id=1, n_frames=128, frame_num=98):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(18, 14), facecolor='white')
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.2)
    
    sr = 22050
    target_frame = f"{frame_num:05d}.png"
    
    # --- 1. Input (Mono Slice Doubled) ---
    source_wav = f"RWAVS/release/{scene_id}/source_syn_re.wav"
    if os.path.exists(source_wav):
        full_source, _ = sf.read(source_wav)
        start = (frame_num - 1) * sr
        end = frame_num * sr
        mono_slice = full_source[start:end]
        input_audio = np.stack([mono_slice, mono_slice], axis=1)
        draw_method_subplot(fig, gs[0, 0], input_audio, sr, "Input (Mono Source)")
    else:
        print(f"Warning: Source wav not found at {source_wav}")
    
    # --- 2. Target View ---
    ax_target = fig.add_subplot(gs[0, 1])
    img_path = f"RWAVS/release/{scene_id}/frames/{target_frame}"
    if os.path.exists(img_path):
        img = Image.open(img_path)
        ax_target.imshow(img)
    ax_target.set_title(f"Target View (Frame {frame_num})", fontsize=16, fontweight='bold', pad=12, color='black')
    ax_target.axis('off')
    
    # --- 3. Our Model ---
    # Path to our results
    wav_ours_frame = f"work_dirs/vggt_experiment_scene_{scene_id}/{frame_num}_pred.wav"
    wav_ours_fallback = f"work_dirs/vggt_experiment_scene_{scene_id}/tmp_pred/1.wav"
    
    if os.path.exists(wav_ours_frame):
        ours_audio, _ = sf.read(wav_ours_frame)
        draw_method_subplot(fig, gs[1, 0], ours_audio, sr, "Our Model")
    else:
        print(f"Warning: Prediction wav not found at {wav_ours_frame}")
        
    # --- 4. Ground Truth ---
    wav_gt_frame = f"work_dirs/vggt_experiment_scene_{scene_id}/{frame_num}_gt.wav"
    wav_gt_fallback = f"work_dirs/vggt_experiment_scene_{scene_id}/tmp_gt/1.wav"
    
    if os.path.exists(wav_gt_frame):
        gt_audio, _ = sf.read(wav_gt_frame)
        draw_method_subplot(fig, gs[1, 1], gt_audio, sr, "Ground Truth")
    else:
        print(f"Warning: GT wav not found at {wav_gt_frame}")

    plt.suptitle("Comparative Analysis: Our Model", fontsize=26, fontweight='bold', y=0.98, color='blue')
    out_name = f"final_analysis_scene_{scene_id}_f{frame_num}.jpg"
    plt.savefig(out_name, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved final comparative analysis to {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=int, default=1)
    parser.add_argument('--frame', type=int, default=98)
    parser.add_argument('--n_frames', type=int, default=128)
    args = parser.parse_args()
    
    plot_final_comparison(args.scene, args.n_frames, args.frame)
