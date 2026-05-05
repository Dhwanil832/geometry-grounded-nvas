import os
import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

def plot_binaural(gt_wav_path, pred_wav_path, scene_name, n_frames, output_path):
    if not os.path.exists(gt_wav_path) or not os.path.exists(pred_wav_path):
        print(f"Error: Files not found for {scene_name}, {n_frames}")
        return

    # Load audio
    gt_audio, sr = sf.read(gt_wav_path)
    pred_audio, _ = sf.read(pred_wav_path)
    
    # Ensure same length and correct shape (N, 2)
    min_len = min(len(gt_audio), len(pred_audio))
    gt_audio = gt_audio[:min_len]
    pred_audio = pred_audio[:min_len]
    
    if gt_audio.shape[1] < 2:
        print(f"Warning: {gt_wav_path} is not binaural.")
        return

    # Create figure with high quality
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(16, 14), facecolor='white')
    plt.suptitle("Binaural Analysis: Our Model", 
                 fontsize=22, fontweight='bold', color='black', y=0.98)
    
    spec_params = {'n_fft': 512, 'win_length': 512, 'hop_length': 128}
    
    # Layout:
    # Row 0: Waveform L (Overlaid)
    # Row 1: Waveform R (Overlaid)
    # Row 2: Spectrogram L (GT | Pred)
    # Row 3: Spectrogram R (GT | Pred)
    
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.1)
    
    # --- 1 & 2: Overlaid Waveforms (Wide) ---
    colors = ['#00e5ff', '#ff00ff'] # Cyan for GT, Magenta for Our Model
    labels = ['Ground Truth', 'Our Model']
    
    for i in range(2):
        ax = fig.add_subplot(gs[i, :]) # Span both columns
        side = "Left" if i == 0 else "Right"
        
        # Calculate pretty limits
        limit = np.max(np.abs(np.concatenate([gt_audio[:, i], pred_audio[:, i]]))) * 1.1
        
        # Plot Pred first so GT is on top
        ax.plot(pred_audio[:, i], color=colors[1], label=labels[1], alpha=0.5, linewidth=0.8)
        ax.plot(gt_audio[:, i], color=colors[0], label=labels[0], alpha=0.8, linewidth=0.8)
        
        ax.set_title(f"{side} Channel Waveform Overlay", fontsize=16, fontweight='bold', color='black', pad=10)
        ax.set_ylim(-limit, limit)
        ax.set_facecolor('#fdfdfd')
        ax.grid(color='black', alpha=0.1)
        if i == 0: ax.legend(loc='upper right', framealpha=0.8, facecolor='white', edgecolor='gray')
        ax.set_ylabel("Amplitude", fontsize=12, color='black')

    # --- 3 & 4: Side-by-Side Spectrograms ---
    for i in range(2): # Channel Loop
        side = "Left" if i == 0 else "Right"
        
        # Compute STFTs for both
        gt_stft = np.abs(librosa.stft(gt_audio[:, i], **spec_params))
        pred_stft = np.abs(librosa.stft(pred_audio[:, i], **spec_params))
        
        # Convert to DB
        gt_db = librosa.amplitude_to_db(gt_stft, ref=np.max)
        pred_db = librosa.amplitude_to_db(pred_stft, ref=np.max)
        
        # Common scale for comparison
        vmin = min(gt_db.min(), pred_db.min())
        vmax = max(gt_db.max(), pred_db.max())
        
        # Plot GT
        ax_gt = fig.add_subplot(gs[i+2, 0])
        librosa.display.specshow(gt_db, sr=sr, **spec_params, x_axis='time', y_axis='linear', 
                                 ax=ax_gt, cmap='viridis', vmin=vmin, vmax=vmax)
        ax_gt.set_title(f"{side} GT Spec", fontsize=14, color='blue') # Darker blue for contrast
        ax_gt.tick_params(colors='black')
        
        # Plot Pred/Our Model
        ax_pred = fig.add_subplot(gs[i+2, 1])
        img = librosa.display.specshow(pred_db, sr=sr, **spec_params, x_axis='time', y_axis='linear', 
                                       ax=ax_pred, cmap='viridis', vmin=vmin, vmax=vmax)
        ax_pred.set_title(f"{side} Our Model Spec", fontsize=14, color='purple') # Darker purple for contrast
        ax_pred.set_yticklabels([]) # Hide Y labels on right side
        ax_pred.set_ylabel("")
        ax_pred.tick_params(colors='black')
        
        # Add colorbar only on the right
        cbar = fig.colorbar(img, ax=[ax_gt, ax_pred], format="%+2.0f dB", pad=0.02)
        cbar.ax.tick_params(labelcolor='black')

    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Generated enhanced plot: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=int, default=1)
    parser.add_argument('--n_frames', type=int, default=128)
    parser.add_argument('--frame', type=int, default=98)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    scene_name = f"vggt_experiment_scene_{args.scene}"
    
    # Try specific frame first, then fallback
    gt_path = f"work_dirs/{scene_name}/{args.n_frames}_gt.wav" # Using n_frames as a stand-in for "best frame" if needed
    if not os.path.exists(gt_path):
        gt_path = f"work_dirs/{scene_name}/{args.frame}_gt.wav"
        
    if not os.path.exists(gt_path):
        gt_path = f"work_dirs/{scene_name}/tmp_gt/1.wav"
        
    pred_path = f"work_dirs/{scene_name}/{args.frame}_pred.wav"
    
    output = args.output if args.output else f"enhanced_binaural_scene_{args.scene}_n{args.n_frames}.jpg"
    plot_binaural(gt_path, pred_path, f"Scene {args.scene}", args.n_frames, output)

if __name__ == "__main__":
    main()
