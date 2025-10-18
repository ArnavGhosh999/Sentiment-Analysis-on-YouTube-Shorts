#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def setup_directories():
    base_dir = Path(__file__).parent
    temp_dir = base_dir / "temp"
    tone_dir = base_dir / "tone_analysis"
    audio_dir = tone_dir / "audio"
    
    for directory in [temp_dir, tone_dir, audio_dir]:
        os.makedirs(directory, exist_ok=True)
    
    viz_dirs = ["waveforms", "features", "heatmaps", "energy"]
    for viz_dir in viz_dirs:
        os.makedirs(audio_dir / viz_dir, exist_ok=True)
    
    return {
        "base": base_dir,
        "temp": temp_dir,
        "tone": tone_dir,
        "audio": audio_dir,
        "waveforms": audio_dir / "waveforms",
        "features": audio_dir / "features",
        "heatmaps": audio_dir / "heatmaps",
        "energy": audio_dir / "energy"
    }

def extract_visual_audio_proxies(video_path):
    """Extract visual features that correlate with audio properties"""
    print(f"Analyzing video for audio-correlated visual features: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    print(f"Video has {total_frames} frames at {fps} FPS, duration: {duration:.2f} seconds")
    
    sample_interval = max(1, int(fps / 10))
    
    frame_data = []
    motion_scores = []
    brightness_values = []
    saturation_values = []
    
    prev_frame = None
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % sample_interval == 0:
            time_point = frame_count / fps
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            brightness = np.mean(hsv[:,:,2])
            brightness_values.append(brightness)
            
            saturation = np.mean(hsv[:,:,1])
            saturation_values.append(saturation)
            
            if prev_frame is not None:
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                
                diff = cv2.absdiff(prev_gray, curr_gray)
                motion_score = np.mean(diff) / 255.0
                
                motion_scores.append({
                    "frame": frame_count,
                    "time": time_point,
                    "motion_score": float(motion_score)
                })
            
            if frame_count % (sample_interval * 30) == 0:
                frame_data.append({
                    "frame": frame_count,
                    "time": time_point,
                    "brightness": float(brightness),
                    "saturation": float(saturation)
                })
            
            prev_frame = frame
            
        frame_count += 1
        
        if frame_count % (sample_interval * 100) == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
    
    cap.release()
    
    if not motion_scores:
        raise ValueError("No motion data extracted from video")
    
    brightness_array = np.array(brightness_values)
    brightness_diff = np.abs(np.diff(brightness_array))
    brightness_diff = np.append(brightness_diff, brightness_diff[-1])
    
    motion_array = np.array([m["motion_score"] for m in motion_scores])
    
    saturation_array = np.array(saturation_values)
    saturation_diff = np.abs(np.diff(saturation_array))
    saturation_diff = np.append(saturation_diff, saturation_diff[-1])
    
    times = np.array([m["time"] for m in motion_scores])
    
    min_length = min(len(brightness_diff), len(motion_array), len(saturation_diff), len(times))
    brightness_diff = brightness_diff[:min_length]
    saturation_diff = saturation_diff[:min_length]
    motion_array = motion_array[:min_length]
    times = times[:min_length]
    
    brightness_array = brightness_array[:min_length]
    saturation_array = saturation_array[:min_length]
    
    from scipy import signal
    peaks, _ = signal.find_peaks(motion_array, distance=int(fps/2))
    if len(peaks) > 1:
        avg_peak_distance = np.mean(np.diff(peaks))
        tempo = 60 / (avg_peak_distance / fps)
    else:
        tempo = 120
    
    return {
        "total_frames": total_frames,
        "fps": fps,
        "duration": duration,
        "times": times,
        "proxy_rms": brightness_diff,
        "proxy_zcr": saturation_diff,
        "proxy_onset": motion_array,
        "brightness": brightness_array,
        "saturation": saturation_array,
        "tempo": tempo,
        "frame_data": frame_data,
        "motion_scores": motion_scores
    }

def create_proxy_waveform(features, dirs):
    print("Creating proxy waveform visualizations...")
    
    times = features["times"]
    proxy_rms = features["proxy_rms"]
    proxy_zcr = features["proxy_zcr"]
    proxy_onset = features["proxy_onset"]
    
    plt.figure(figsize=(14, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(times, proxy_rms, color='blue')
    plt.title('Brightness Variation (Audio Energy Proxy)')
    plt.ylabel('Variation')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(times, proxy_zcr, color='green')
    plt.title('Saturation Variation (Audio Texture Proxy)')
    plt.ylabel('Variation')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(times, proxy_onset, color='red')
    plt.title('Motion Intensity (Audio Onset Proxy)')
    plt.xlabel('Time (s)')
    plt.ylabel('Intensity')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(dirs["waveforms"] / "proxy_waveform.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(14, 6))
    
    scaler = MinMaxScaler()
    rms_norm = scaler.fit_transform(proxy_rms.reshape(-1, 1)).flatten()
    zcr_norm = scaler.fit_transform(proxy_zcr.reshape(-1, 1)).flatten()
    onset_norm = scaler.fit_transform(proxy_onset.reshape(-1, 1)).flatten()
    
    plt.plot(times, rms_norm, color='blue', label='Energy Proxy', alpha=0.7)
    plt.plot(times, zcr_norm, color='green', label='Texture Proxy', alpha=0.7)
    plt.plot(times, onset_norm, color='red', label='Onset Proxy', alpha=0.7)
    
    plt.title('Combined Audio Proxy Features')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(dirs["waveforms"] / "combined_proxy.png", dpi=300)
    plt.close()

def create_energy_visualizations(features, dirs):
    print("Creating energy visualizations...")
    
    times = features["times"]
    proxy_rms = features["proxy_rms"]
    motion = features["proxy_onset"]
    
    plt.figure(figsize=(14, 6))
    plt.plot(times, proxy_rms, color='blue', linewidth=2)
    plt.fill_between(times, proxy_rms, alpha=0.5, color='blue')
    plt.title('Visual Energy Contour (Audio Energy Proxy)')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy Proxy')
    plt.grid(True, alpha=0.3)
    
    from scipy import signal
    peaks, _ = signal.find_peaks(motion, height=np.mean(motion) + np.std(motion))
    peak_times = [times[peak] for peak in peaks if peak < len(times)]
    
    if len(peak_times) > 0:
        plt.vlines(peak_times, 0, proxy_rms.max(), color='r', alpha=0.7, 
                  linewidth=1, label='Activity Peaks')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(dirs["energy"] / "energy_contour.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.hist(proxy_rms, bins=50, alpha=0.7, color='blue')
    plt.axvline(np.mean(proxy_rms), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(proxy_rms):.4f}')
    plt.axvline(np.median(proxy_rms), color='green', linestyle='dashed', linewidth=2, label=f'Median: {np.median(proxy_rms):.4f}')
    plt.title('Energy Proxy Distribution')
    plt.xlabel('Energy Proxy')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(dirs["energy"] / "energy_histogram.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(14, 6))
    
    n_segments = 20
    segment_size = len(proxy_rms) // n_segments
    
    if segment_size > 0:
        rms_reshaped = np.zeros((n_segments, segment_size))
        
        for i in range(n_segments):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size
            if end_idx <= len(proxy_rms):
                rms_reshaped[i, :] = proxy_rms[start_idx:end_idx]
        
        extent = [0, features["duration"], 0, n_segments]
        
        plt.imshow(rms_reshaped, aspect='auto', origin='lower', 
                  cmap='inferno', extent=extent)
        plt.colorbar(label='Energy Proxy')
        plt.title('Energy Intensity Heatmap')
        plt.xlabel('Time (s)')
        plt.ylabel('Segment')
        plt.tight_layout()
        plt.savefig(dirs["energy"] / "energy_heatmap.png", dpi=300)
        plt.close()

def create_feature_visualizations(features, dirs):
    print("Creating feature visualizations...")
    
    times = features["times"]
    brightness = features["brightness"]
    saturation = features["saturation"]
    motion = features["proxy_onset"]
    
    plt.figure(figsize=(14, 4))
    plt.plot(times, brightness, color='yellow', linewidth=2)
    plt.title('Brightness Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Brightness Value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(dirs["features"] / "brightness.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(14, 4))
    plt.plot(times, saturation, color='magenta', linewidth=2)
    plt.title('Saturation Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Saturation Value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(dirs["features"] / "saturation.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(14, 4))
    plt.plot(times, motion, color='red', linewidth=2)
    plt.title('Motion Intensity Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Motion Intensity')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(dirs["features"] / "motion.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(14, 10))
    
    scaler = MinMaxScaler()
    brightness_norm = scaler.fit_transform(brightness.reshape(-1, 1)).flatten()
    saturation_norm = scaler.fit_transform(saturation.reshape(-1, 1)).flatten()
    motion_norm = scaler.fit_transform(motion.reshape(-1, 1)).flatten()
    
    plt.subplot(3, 1, 1)
    plt.plot(times, brightness_norm, color='yellow', linewidth=2)
    plt.title('Brightness (Normalized)')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(times, saturation_norm, color='magenta', linewidth=2)
    plt.title('Saturation (Normalized)')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(times, motion_norm, color='red', linewidth=2)
    plt.title('Motion (Normalized)')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(dirs["features"] / "combined_features.png", dpi=300)
    plt.close()

def create_heatmap_visualizations(features, dirs):
    print("Creating heatmap visualizations...")
    
    times = features["times"]
    
    scaler = MinMaxScaler()
    
    feature_data = {
        "Brightness": scaler.fit_transform(features["brightness"].reshape(-1, 1)).flatten(),
        "Saturation": scaler.fit_transform(features["saturation"].reshape(-1, 1)).flatten(),
        "Motion": scaler.fit_transform(features["proxy_onset"].reshape(-1, 1)).flatten(),
        "Energy Proxy": scaler.fit_transform(features["proxy_rms"].reshape(-1, 1)).flatten(),
        "Texture Proxy": scaler.fit_transform(features["proxy_zcr"].reshape(-1, 1)).flatten()
    }
    
    time_segments = []
    
    segment_size = len(times) // 20
    
    if segment_size > 0:
        for i in range(0, len(times), segment_size):
            end_idx = min(i + segment_size, len(times))
            segment_time = f"{times[i]:.1f}-{times[end_idx-1]:.1f}s"
            
            segment_data = {}
            for feature_name, feature_values in feature_data.items():
                segment_data[feature_name] = np.mean(feature_values[i:end_idx])
            
            segment_data["Time"] = segment_time
            time_segments.append(segment_data)
        
        df = pd.DataFrame(time_segments)
        df = df.set_index("Time")
        
        plt.figure(figsize=(14, 8))
        sns.heatmap(df, cmap="YlOrRd", annot=True, fmt=".2f", linewidths=.5)
        plt.title("Visual Features Over Time")
        plt.tight_layout()
        plt.savefig(dirs["heatmaps"] / "feature_heatmap.png", dpi=300)
        plt.close()
    
    plt.figure(figsize=(10, 8))
    
    corr_data = pd.DataFrame(feature_data)
    correlation = corr_data.corr()
    
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    
    sns.heatmap(correlation, mask=mask, cmap="coolwarm", annot=True, 
               fmt=".2f", square=True, linewidths=.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(dirs["heatmaps"] / "correlation_heatmap.png", dpi=300)
    plt.close()

def map_features_to_emotion(features):
    print("Mapping visual features to emotions...")
    
    avg_brightness = np.mean(features["brightness"])
    avg_saturation = np.mean(features["saturation"])
    avg_motion = np.mean(features["proxy_onset"])
    tempo = features["tempo"]
    
    norm_brightness = min(avg_brightness / 200, 1.0)
    norm_saturation = min(avg_saturation / 200, 1.0)
    norm_motion = min(avg_motion / 0.2, 1.0)
    norm_tempo = min(tempo / 180, 1.0)
    
    valence = (norm_brightness * 0.7 + norm_saturation * 0.3)
    arousal = (norm_motion * 0.6 + norm_tempo * 0.4)
    tension = ((1 - norm_brightness) * 0.4 + norm_motion * 0.6)
    
    if arousal > 0.7:
        if valence > 0.7:
            primary_emotion = "excitement"
        elif valence > 0.4:
            primary_emotion = "happiness"
        else:
            primary_emotion = "anger"
    elif arousal > 0.4:
        if valence > 0.6:
            primary_emotion = "contentment"
        elif valence > 0.3:
            primary_emotion = "neutral"
        else:
            primary_emotion = "fear"
    else:
        if valence > 0.5:
            primary_emotion = "calmness"
        else:
            primary_emotion = "sadness"
    
    times = features["times"]
    brightness = features["brightness"]
    saturation = features["saturation"]
    motion = features["proxy_onset"]
    
    segment_duration = 5
    num_segments = int(np.ceil(features["duration"] / segment_duration))
    
    emotion_segments = []
    
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, features["duration"])
        
        indices = np.where((times >= start_time) & (times < end_time))[0]
        
        if len(indices) == 0:
            continue
        
        seg_brightness = np.mean(brightness[indices])
        seg_saturation = np.mean(saturation[indices])
        seg_motion = np.mean(motion[indices])
        
        seg_norm_brightness = min(seg_brightness / 200, 1.0)
        seg_norm_saturation = min(seg_saturation / 200, 1.0)
        seg_norm_motion = min(seg_motion / 0.2, 1.0)
        
        seg_valence = (seg_norm_brightness * 0.7 + seg_norm_saturation * 0.3)
        seg_arousal = (seg_norm_motion * 0.6 + norm_tempo * 0.4)
        seg_tension = ((1 - seg_norm_brightness) * 0.4 + seg_norm_motion * 0.6)
        
        if seg_arousal > 0.7:
            if seg_valence > 0.7:
                seg_emotion = "excitement"
            elif seg_valence > 0.4:
                seg_emotion = "happiness"
            else:
                seg_emotion = "anger"
        elif seg_arousal > 0.4:
            if seg_valence > 0.6:
                seg_emotion = "contentment"
            elif seg_valence > 0.3:
                seg_emotion = "neutral"
            else:
                seg_emotion = "fear"
        else:
            if seg_valence > 0.5:
                seg_emotion = "calmness"
            else:
                seg_emotion = "sadness"
        
        emotion_segments.append({
            "segment": i,
            "start_time": float(start_time),
            "end_time": float(end_time),
            "arousal": float(seg_arousal),
            "valence": float(seg_valence),
            "tension": float(seg_tension),
            "primary_emotion": seg_emotion
        })
    
    return {
        "overall_arousal": float(arousal),
        "overall_valence": float(valence),
        "overall_tension": float(tension),
        "primary_emotion": primary_emotion,
        "segments": emotion_segments
    }

def create_emotion_distribution_pie(emotion_data, dirs):
    """Create pie chart with large readable text and legend"""
    print("Creating emotion distribution pie chart...")
    
    segments = emotion_data["segments"]
    
    emotion_counts = {}
    for segment in segments:
        emotion = segment["primary_emotion"]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    total_segments = len(segments)
    emotion_distribution = {emotion: count/total_segments for emotion, count in emotion_counts.items()}
    
    emotion_colors = {
        "excitement": '#FF1744',
        "happiness": '#FFA500',
        "contentment": '#FFD700',
        "crowd_excitement": '#00DD00',
        "cheerfulness": '#00FF00',
        "calmness": '#0099FF',
        "serenity": '#00DDFF',
        "anxiety": '#9900FF',
        "neutral": '#888888',
        "sadness": '#0033FF',
        "fear": '#6600FF',
        "anger": '#CC0000',
        "elation": '#FF0099',
        "distress": '#FF3300',
        "boredom": '#666666',
        "depression": '#003366'
    }
    
    sorted_emotions = sorted(emotion_distribution.items(), key=lambda x: x[1], reverse=True)
    labels = [e[0] for e in sorted_emotions]
    sizes = [e[1] * 100 for e in sorted_emotions]
    colors = [emotion_colors.get(label, '#CCCCCC') for label in labels]
    
    fig, ax = plt.subplots(figsize=(18, 14))
    
    # Create pie without labels first
    wedges, texts, autotexts = ax.pie(
        sizes,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 22, 'weight': 'bold'},
        wedgeprops=dict(edgecolor='black', linewidth=4)
    )
    
    # Set percentage text color based on background
    for autotext, color in zip(autotexts, colors):
        autotext.set_fontsize(22)
        autotext.set_weight('bold')
        # Use dark text on light backgrounds, light text on dark
        if color in ['#FFD700', '#FFA500', '#00DD00', '#00FF00', '#00DDFF', '#FFD700']:
            autotext.set_color('black')
        else:
            autotext.set_color('white')
    
    # Create legend with all emotions
    legend_elements = []
    for label, size, color in zip(labels, sizes, colors):
        legend_elements.append(
            plt.Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor=color,
                markersize=18,
                label=f'{label.capitalize()}: {size:.1f}%',
                markeredgecolor='black',
                markeredgewidth=2
            )
        )
    
    ax.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=16,
        frameon=True,
        fancybox=True,
        shadow=True,
        title='Emotion Categories',
        title_fontsize=18
    )
    
    ax.set_title('Emotion Distribution Across Video', fontsize=26, weight='bold', pad=30)
    
    plt.tight_layout()
    plt.savefig(dirs["audio"] / "emotion_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Pie chart saved successfully!")

def create_emotion_timeline_enhanced(emotion_data, dirs):
    """Create enhanced emotion timeline with vivid styling"""
    print("Creating enhanced emotion timeline...")
    
    segments = emotion_data["segments"]
    times = [(s["start_time"] + s["end_time"]) / 2 for s in segments]
    arousal = [s["arousal"] for s in segments]
    valence = [s["valence"] for s in segments]
    tension = [s["tension"] for s in segments]
    
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(18, 8))
    
    ax.plot(times, arousal, linewidth=4, label='Arousal', color='#FF6B6B', alpha=0.9, marker='o', markersize=5)
    ax.plot(times, valence, linewidth=4, label='Valence', color='#2CC5E5', alpha=0.9, marker='s', markersize=5)
    ax.plot(times, tension, linewidth=4, label='Tension', color='#FFB703', alpha=0.9, marker='^', markersize=5)
    
    ax.fill_between(times, arousal, alpha=0.2, color='#FF6B6B')
    ax.fill_between(times, valence, alpha=0.2, color='#2CC5E5')
    ax.fill_between(times, tension, alpha=0.2, color='#FFB703')
    
    ax.set_xlabel('Time (seconds)', fontsize=15, weight='bold')
    ax.set_ylabel('Intensity', fontsize=15, weight='bold')
    ax.set_title('Temporal Evolution of Emotional Dimensions', fontsize=18, weight='bold', pad=20)
    ax.legend(fontsize=14, loc='upper right', frameon=True, shadow=True, fancybox=True)
    ax.grid(True, alpha=0.3, linewidth=1.5)
    ax.set_ylim(0, 1.1)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(dirs["audio"] / "emotion_timeline_enhanced.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_emotion_heatmap_advanced(emotion_data, dirs):
    """Create advanced heatmap with vivid colors"""
    print("Creating advanced emotion heatmap...")
    
    segments = emotion_data["segments"]
    
    emotion_matrix = []
    time_labels = []
    
    for segment in segments:
        time_labels.append(f"{segment['start_time']:.1f}s")
        emotion_matrix.append({
            'Arousal': segment['arousal'],
            'Valence': segment['valence'],
            'Tension': segment['tension']
        })
    
    df = pd.DataFrame(emotion_matrix, index=time_labels)
    
    fig, ax = plt.subplots(figsize=(18, 8))
    sns.heatmap(df.T, cmap='RdYlGn', annot=True, fmt='.2f', 
                cbar_kws={'label': 'Intensity', 'shrink': 0.8}, 
                linewidths=1, ax=ax, linecolor='white',
                vmin=0, vmax=1, cbar=True, annot_kws={'fontsize': 10, 'weight': 'bold'})
    
    ax.set_title('Emotion Dimensions Heatmap Over Time', fontsize=18, weight='bold', pad=20)
    ax.set_xlabel('Time Segments', fontsize=14, weight='bold')
    ax.set_ylabel('Emotional Dimensions', fontsize=14, weight='bold')
    ax.tick_params(labelsize=11)
    
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(dirs["audio"] / "emotion_heatmap_advanced.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_emotion_radar_chart(emotion_data, dirs):
    """Create vivid radar chart for emotional profile"""
    print("Creating emotion radar chart...")
    
    categories = ['Arousal', 'Valence', 'Tension']
    values = [
        emotion_data['overall_arousal'],
        emotion_data['overall_valence'],
        emotion_data['overall_tension']
    ]
    values += values[:1]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=4, color='#FF6B6B', markersize=12, markeredgewidth=2, markeredgecolor='white')
    ax.fill(angles, values, alpha=0.35, color='#FF6B6B')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=14, weight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=11, weight='bold')
    ax.set_title(f'Overall Emotional Profile\n{emotion_data["primary_emotion"].upper()}', 
                 fontsize=16, weight='bold', pad=30)
    ax.grid(True, linewidth=1.5)
    
    plt.tight_layout()
    plt.savefig(dirs["audio"] / "emotion_radar_chart.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_emotion_distribution_violin(emotion_data, dirs):
    """Create vivid violin plot for emotion distributions"""
    print("Creating emotion distribution violin plot...")
    
    segments = emotion_data["segments"]
    
    data = {
        'Arousal': [s['arousal'] for s in segments],
        'Valence': [s['valence'] for s in segments],
        'Tension': [s['tension'] for s in segments]
    }
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    parts = ax.violinplot([data['Arousal'], data['Valence'], data['Tension']], 
                          positions=[1, 2, 3], showmeans=True, showmedians=True, widths=0.7)
    
    colors_violin = ['#FF6B6B', '#2CC5E5', '#FFB703']
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_violin[i])
        pc.set_alpha(0.8)
        pc.set_edgecolor('black')
        pc.set_linewidth(2)
    
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(2.5)
    
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Arousal', 'Valence', 'Tension'], fontsize=14, weight='bold')
    ax.set_ylabel('Intensity', fontsize=14, weight='bold')
    ax.set_title('Distribution of Emotional Dimensions', fontsize=16, weight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y', linewidth=1.5)
    ax.set_ylim(0, 1.1)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(dirs["audio"] / "emotion_violin_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_emotion_transition_network(emotion_data, dirs):
    """Create vivid emotion transition diagram"""
    print("Creating emotion transition diagram...")
    
    segments = emotion_data["segments"]
    
    transitions = {}
    for i in range(len(segments) - 1):
        current = segments[i]['primary_emotion']
        next_emotion = segments[i + 1]['primary_emotion']
        key = f"{current} → {next_emotion}"
        transitions[key] = transitions.get(key, 0) + 1
    
    sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:12]
    labels, counts = zip(*sorted_transitions)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    colors_list = sns.color_palette("husl", len(labels))
    bars = ax.barh(range(len(labels)), counts, color=colors_list, edgecolor='black', linewidth=2.5, height=0.7)
    
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=12, weight='bold')
    ax.set_xlabel('Frequency', fontsize=14, weight='bold')
    ax.set_title('Top 12 Emotion Transitions', fontsize=16, weight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x', linewidth=1.5)
    ax.tick_params(labelsize=12)
    
    for i, (bar, count) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{int(count)}',
                ha='left', va='center', fontsize=12, weight='bold')
    
    plt.tight_layout()
    plt.savefig(dirs["audio"] / "emotion_transitions.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_emotional_visualizations(emotion_data, features, dirs):
    print("Creating emotional analysis visualizations...")
    
    create_emotion_distribution_pie(emotion_data, dirs)
    create_emotion_timeline_enhanced(emotion_data, dirs)
    create_emotion_heatmap_advanced(emotion_data, dirs)
    create_emotion_radar_chart(emotion_data, dirs)
    create_emotion_distribution_violin(emotion_data, dirs)
    create_emotion_transition_network(emotion_data, dirs)
    
    segments = emotion_data["segments"]
    times = [(s["start_time"] + s["end_time"]) / 2 for s in segments]
    valence = [s["valence"] for s in segments]
    arousal = [s["arousal"] for s in segments]
    emotions = [s["primary_emotion"] for s in segments]
    
    plt.figure(figsize=(12, 10))
    
    emotion_colors = {
        "excitement": '#E63946',
        "happiness": '#F77F00',
        "contentment": '#FCBF49',
        "calmness": '#003D5B',
        "neutral": '#616161',
        "sadness": '#1565C0',
        "fear": '#6A1B9A',
        "anger": '#C62828'
    }
    
    colors = [emotion_colors.get(emotion, '#999999') for emotion in emotions]
    
    plt.scatter(valence, arousal, c=colors, s=200, alpha=0.8, edgecolors='black', linewidth=2)
    
    plt.text(0.25, 0.9, "High Energy\nNegative", ha='center', fontsize=13, weight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6, pad=0.7))
    plt.text(0.75, 0.9, "High Energy\nPositive", ha='center', fontsize=13, weight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6, pad=0.7))
    plt.text(0.25, 0.1, "Low Energy\nNegative", ha='center', fontsize=13, weight='bold',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.6, pad=0.7))
    plt.text(0.75, 0.1, "Low Energy\nPositive", ha='center', fontsize=13, weight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6, pad=0.7))
    
    legend_elements = []
    for emotion, color in emotion_colors.items():
        if emotion in emotions:
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                          markersize=14, label=emotion.capitalize(), 
                          markeredgecolor='black', markeredgewidth=1.5)
            )
    
    plt.legend(handles=legend_elements, fontsize=14, loc='upper left', 
               frameon=True, shadow=True, fancybox=True, title='Emotions', title_fontsize=15)
    
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.3, linewidth=2.5)
    plt.axvline(x=0.5, color='k', linestyle='--', alpha=0.3, linewidth=2.5)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Valence (Negative ← → Positive)', fontsize=14, weight='bold')
    plt.ylabel('Arousal (Low ← → High)', fontsize=14, weight='bold')
    plt.title('2D Emotional Space: Valence vs Arousal', fontsize=16, weight='bold', pad=20)
    plt.grid(True, alpha=0.2, linewidth=1.5)
    plt.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(dirs["audio"] / "valence_arousal.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_analysis_data(features, emotion_data, dirs):
    print("Saving analysis data...")
    
    summary = {
        "duration": features["duration"],
        "tempo": float(features["tempo"]),
        "overall_emotion": emotion_data["primary_emotion"],
        "emotional_dimensions": {
            "arousal": emotion_data["overall_arousal"],
            "valence": emotion_data["overall_valence"],
            "tension": emotion_data["overall_tension"]
        },
        "average_features": {
            "brightness": float(np.mean(features["brightness"])),
            "saturation": float(np.mean(features["saturation"])),
            "motion": float(np.mean(features["proxy_onset"]))
        },
        "emotion_segments": emotion_data["segments"]
    }
    
    summary_path = dirs["audio"] / "visual_audio_analysis.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    
    text_summary = [
        "VISUAL-AUDIO ANALYSIS SUMMARY",
        "------------------------------",
        f"Duration: {features['duration']:.2f} seconds",
        f"Tempo: {features['tempo']:.1f} BPM",
        f"",
        f"Overall Emotional Tone: {emotion_data['primary_emotion'].capitalize()}",
        f"Emotional Dimensions:",
        f"  - Arousal: {emotion_data['overall_arousal']:.2f}",
        f"  - Valence: {emotion_data['overall_valence']:.2f}",
        f"  - Tension: {emotion_data['overall_tension']:.2f}",
        f"",
        f"Emotional Segments:",
    ]
    
    emotion_peaks = {}
    for segment in emotion_data["segments"]:
        emotion = segment["primary_emotion"]
        if emotion not in emotion_peaks:
            emotion_peaks[emotion] = []
        
        emotion_peaks[emotion].append((segment["arousal"] + segment["valence"])/2)
    
    text_summary.append("Peak Emotional Moments:")
    for emotion, intensities in emotion_peaks.items():
        if len(intensities) > 0:
            peak_idx = np.argmax(intensities)
            peak_segment = [s for s in emotion_data["segments"] if s["primary_emotion"] == emotion][peak_idx]
            text_summary.append(f"  - {emotion.capitalize()}: {peak_segment['start_time']:.1f}s - {peak_segment['end_time']:.1f}s")
    
    plt.figure(figsize=(12, 8))
    plt.text(0.5, 0.5, "\n".join(text_summary), 
             horizontalalignment='center',
             fontsize=14, 
             linespacing=1.5)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(dirs["audio"] / "text_summary.png", dpi=300)
    plt.close()
    
    return summary

def main():
    try:
        print("\n*** VISUAL-BASED AUDIO ANALYSIS ***\n")
        
        dirs = setup_directories()
        
        video_path = dirs["temp"] / "ftrTyduMBFc.mp4"
        
        if not video_path.exists():
            print(f"Video not found at {video_path}")
            print("Please ensure the video exists at that location")
            sys.exit(1)
        
        features = extract_visual_audio_proxies(video_path)
        
        create_proxy_waveform(features, dirs)
        create_feature_visualizations(features, dirs)
        create_energy_visualizations(features, dirs)
        create_heatmap_visualizations(features, dirs)
        
        emotion_data = map_features_to_emotion(features)
        
        create_emotional_visualizations(emotion_data, features, dirs)
        
        summary = save_analysis_data(features, emotion_data, dirs)
        
        print("\nVisual-based audio analysis complete!")
        print(f"Results saved to: {dirs['audio']}")
        print(f"\nOverall emotion detected: {summary['overall_emotion'].upper()}")
        print(f"Arousal: {summary['emotional_dimensions']['arousal']:.2f}")
        print(f"Valence: {summary['emotional_dimensions']['valence']:.2f}")
        print(f"Tension: {summary['emotional_dimensions']['tension']:.2f}")
        
        print("\nPeak emotional moments:")
        emotion_peaks = {}
        for segment in emotion_data["segments"]:
            emotion = segment["primary_emotion"]
            if emotion not in emotion_peaks:
                emotion_peaks[emotion] = []
            
            emotion_peaks[emotion].append((segment["arousal"] + segment["valence"])/2)
        
        for emotion, intensities in emotion_peaks.items():
            if len(intensities) > 0:
                peak_idx = np.argmax(intensities)
                peak_segment = [s for s in emotion_data["segments"] if s["primary_emotion"] == emotion][peak_idx]
                print(f"  - {emotion.capitalize()}: {peak_segment['start_time']:.1f}s - {peak_segment['end_time']:.1f}s")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()