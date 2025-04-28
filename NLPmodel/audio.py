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
    
    # Create subdirectories for visualizations
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
    
    # Sample frames at regular intervals
    sample_interval = max(1, int(fps / 10))  # 10 samples per second
    
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
            # Calculate time point
            time_point = frame_count / fps
            
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Calculate brightness and saturation (proxy for audio features)
            brightness = np.mean(hsv[:,:,2])
            brightness_values.append(brightness)
            
            saturation = np.mean(hsv[:,:,1])
            saturation_values.append(saturation)
            
            # Calculate motion if we have a previous frame
            if prev_frame is not None:
                # Convert to grayscale
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate frame difference
                diff = cv2.absdiff(prev_gray, curr_gray)
                motion_score = np.mean(diff) / 255.0
                
                motion_scores.append({
                    "frame": frame_count,
                    "time": time_point,
                    "motion_score": float(motion_score)
                })
            
            # Save sample frame
            if frame_count % (sample_interval * 30) == 0:
                frame_data.append({
                    "frame": frame_count,
                    "time": time_point,
                    "brightness": float(brightness),
                    "saturation": float(saturation)
                })
            
            # Update previous frame
            prev_frame = frame
            
        frame_count += 1
        
        # Print progress
        if frame_count % (sample_interval * 100) == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
    
    # Release resources
    cap.release()
    
    # Make sure we have data
    if not motion_scores:
        raise ValueError("No motion data extracted from video")
    
    # Calculate audio-like features from visual data
    
    # 1. Brightness variation as proxy for audio energy (RMS)
    brightness_array = np.array(brightness_values)
    brightness_diff = np.abs(np.diff(brightness_array))
    brightness_diff = np.append(brightness_diff, brightness_diff[-1])  # Pad to match length
    
    # 2. Motion as proxy for audio activity/onset
    motion_array = np.array([m["motion_score"] for m in motion_scores])
    
    # 3. Saturation variation as proxy for audio richness/timbre
    saturation_array = np.array(saturation_values)
    saturation_diff = np.abs(np.diff(saturation_array))
    saturation_diff = np.append(saturation_diff, saturation_diff[-1])  # Pad to match length
    
    # Create times array
    times = np.array([m["time"] for m in motion_scores])
    
    # Make sure all arrays have the same length
    min_length = min(len(brightness_diff), len(motion_array), len(saturation_diff), len(times))
    brightness_diff = brightness_diff[:min_length]
    saturation_diff = saturation_diff[:min_length]
    motion_array = motion_array[:min_length]
    times = times[:min_length]
    
    # Ensure brightness and saturation arrays match
    brightness_array = brightness_array[:min_length]
    saturation_array = saturation_array[:min_length]
    
    # Calculate "tempo" based on motion peaks
    from scipy import signal
    peaks, _ = signal.find_peaks(motion_array, distance=int(fps/2))
    if len(peaks) > 1:
        avg_peak_distance = np.mean(np.diff(peaks))
        tempo = 60 / (avg_peak_distance / fps)
    else:
        tempo = 120  # Default tempo
    
    return {
        "total_frames": total_frames,
        "fps": fps,
        "duration": duration,
        "times": times,
        "proxy_rms": brightness_diff,  # Brightness variation as proxy for audio energy
        "proxy_zcr": saturation_diff,  # Saturation variation as proxy for audio texture
        "proxy_onset": motion_array,   # Motion as proxy for audio onsets
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
    
    # Verify all arrays have the same length
    print(f"Verification - Array lengths: times={len(times)}, proxy_rms={len(proxy_rms)}, " 
          f"proxy_zcr={len(proxy_zcr)}, proxy_onset={len(proxy_onset)}")
    
    # Basic waveform-like visualization
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
    
    # Combined visualization
    plt.figure(figsize=(14, 6))
    
    # Normalize for better comparison
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
    
    # Energy contour plot
    plt.figure(figsize=(14, 6))
    plt.plot(times, proxy_rms, color='blue', linewidth=2)
    plt.fill_between(times, proxy_rms, alpha=0.5, color='blue')
    plt.title('Visual Energy Contour (Audio Energy Proxy)')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy Proxy')
    plt.grid(True, alpha=0.3)
    
    # Add onset markers
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
    
    # Energy histogram
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
    
    # Create energy heatmap (time vs intensity)
    plt.figure(figsize=(14, 6))
    
    # Reshape RMS to create a 2D representation
    n_segments = 20  # Number of vertical segments
    segment_size = len(proxy_rms) // n_segments
    
    if segment_size > 0:  # Make sure we have enough data
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
    
    # Brightness
    plt.figure(figsize=(14, 4))
    plt.plot(times, brightness, color='yellow', linewidth=2)
    plt.title('Brightness Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Brightness Value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(dirs["features"] / "brightness.png", dpi=300)
    plt.close()
    
    # Saturation
    plt.figure(figsize=(14, 4))
    plt.plot(times, saturation, color='magenta', linewidth=2)
    plt.title('Saturation Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Saturation Value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(dirs["features"] / "saturation.png", dpi=300)
    plt.close()
    
    # Motion
    plt.figure(figsize=(14, 4))
    plt.plot(times, motion, color='red', linewidth=2)
    plt.title('Motion Intensity Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Motion Intensity')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(dirs["features"] / "motion.png", dpi=300)
    plt.close()
    
    # Combined features plot
    plt.figure(figsize=(14, 10))
    
    # Normalize features for comparison
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
    
    # Multi-feature heatmap
    times = features["times"]
    
    # Prepare data for heatmap - normalize all features
    scaler = MinMaxScaler()
    
    # Select features for the heatmap
    feature_data = {
        "Brightness": scaler.fit_transform(features["brightness"].reshape(-1, 1)).flatten(),
        "Saturation": scaler.fit_transform(features["saturation"].reshape(-1, 1)).flatten(),
        "Motion": scaler.fit_transform(features["proxy_onset"].reshape(-1, 1)).flatten(),
        "Energy Proxy": scaler.fit_transform(features["proxy_rms"].reshape(-1, 1)).flatten(),
        "Texture Proxy": scaler.fit_transform(features["proxy_zcr"].reshape(-1, 1)).flatten()
    }
    
    # Create DataFrame for heatmap
    time_segments = []
    
    # Create 20 time segments for better visualization
    segment_size = len(times) // 20
    
    if segment_size > 0:
        for i in range(0, len(times), segment_size):
            end_idx = min(i + segment_size, len(times))
            segment_time = f"{times[i]:.1f}-{times[end_idx-1]:.1f}s"
            
            # Calculate average of each feature in this segment
            segment_data = {}
            for feature_name, feature_values in feature_data.items():
                segment_data[feature_name] = np.mean(feature_values[i:end_idx])
            
            segment_data["Time"] = segment_time
            time_segments.append(segment_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(time_segments)
        df = df.set_index("Time")
        
        # Create heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(df, cmap="YlOrRd", annot=True, fmt=".2f", linewidths=.5)
        plt.title("Visual Features Over Time")
        plt.tight_layout()
        plt.savefig(dirs["heatmaps"] / "feature_heatmap.png", dpi=300)
        plt.close()
    
    # Create a correlation heatmap
    plt.figure(figsize=(10, 8))
    
    # Create correlation matrix
    corr_data = pd.DataFrame(feature_data)
    correlation = corr_data.corr()
    
    # Generate mask for upper triangle
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    
    # Create heatmap
    sns.heatmap(correlation, mask=mask, cmap="coolwarm", annot=True, 
               fmt=".2f", square=True, linewidths=.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(dirs["heatmaps"] / "correlation_heatmap.png", dpi=300)
    plt.close()

def map_features_to_emotion(features):
    print("Mapping visual features to emotions...")
    
    # Extract average values
    avg_brightness = np.mean(features["brightness"])
    avg_saturation = np.mean(features["saturation"])
    avg_motion = np.mean(features["proxy_onset"])
    tempo = features["tempo"]
    
    # Normalize values
    norm_brightness = min(avg_brightness / 200, 1.0)
    norm_saturation = min(avg_saturation / 200, 1.0)
    norm_motion = min(avg_motion / 0.2, 1.0)
    norm_tempo = min(tempo / 180, 1.0)
    
    # Calculate emotional dimensions
    # Valence (positivity) primarily from brightness
    valence = (norm_brightness * 0.7 + norm_saturation * 0.3)
    
    # Arousal (energy) primarily from motion and tempo
    arousal = (norm_motion * 0.6 + norm_tempo * 0.4)
    
    # Tension from inverse of brightness and high motion
    tension = ((1 - norm_brightness) * 0.4 + norm_motion * 0.6)
    
    # Map dimensions to categorical emotions
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
    
    # Segment-level emotions
    times = features["times"]
    brightness = features["brightness"]
    saturation = features["saturation"]
    motion = features["proxy_onset"]
    
    segment_duration = 5  # seconds
    num_segments = int(np.ceil(features["duration"] / segment_duration))
    
    emotion_segments = []
    
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, features["duration"])
        
        # Find indices within this time segment
        indices = np.where((times >= start_time) & (times < end_time))[0]
        
        if len(indices) == 0:
            continue
        
        # Calculate segment-specific values
        seg_brightness = np.mean(brightness[indices])
        seg_saturation = np.mean(saturation[indices])
        seg_motion = np.mean(motion[indices])
        
        # Normalize
        seg_norm_brightness = min(seg_brightness / 200, 1.0)
        seg_norm_saturation = min(seg_saturation / 200, 1.0)
        seg_norm_motion = min(seg_motion / 0.2, 1.0)
        
        # Calculate segment emotional dimensions
        seg_valence = (seg_norm_brightness * 0.7 + seg_norm_saturation * 0.3)
        seg_arousal = (seg_norm_motion * 0.6 + norm_tempo * 0.4)  # Use global tempo
        seg_tension = ((1 - seg_norm_brightness) * 0.4 + seg_norm_motion * 0.6)
        
        # Map to categorical emotion
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

def create_emotional_visualizations(emotion_data, features, dirs):
    print("Creating emotional analysis visualizations...")
    
    # Emotion time series
    segments = emotion_data["segments"]
    times = [(s["start_time"] + s["end_time"]) / 2 for s in segments]
    arousal = [s["arousal"] for s in segments]
    valence = [s["valence"] for s in segments]
    tension = [s["tension"] for s in segments]
    emotions = [s["primary_emotion"] for s in segments]
    
    # Emotion timeline
    plt.figure(figsize=(14, 8))
    plt.plot(times, arousal, 'r-', label='Arousal (Energy)', linewidth=2)
    plt.plot(times, valence, 'g-', label='Valence (Positivity)', linewidth=2)
    plt.plot(times, tension, 'b-', label='Tension', linewidth=2)
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Intensity', fontsize=12)
    plt.title('Emotional Dimensions Over Time', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.ylim(0, 1.1)
    
    # Add emotion labels
    for time, emotion in zip(times, emotions):
        plt.text(time, 1.05, emotion, rotation=45, ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(dirs["audio"] / "emotion_timeline.png", dpi=300)
    plt.close()
    
    # Emotion distribution
    emotion_counts = {}
    for segment in segments:
        emotion = segment["primary_emotion"]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    total_segments = len(segments)
    emotion_distribution = {emotion: count/total_segments for emotion, count in emotion_counts.items()}
    
    # Pie chart of emotions
    plt.figure(figsize=(10, 8))
    labels = list(emotion_distribution.keys())
    sizes = list(emotion_distribution.values())
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
            shadow=True, counterclock=False)
    plt.axis('equal')
    plt.title('Emotion Distribution', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(dirs["audio"] / "emotion_distribution.png", dpi=300)
    plt.close()
    
    # Valence-Arousal plot
    plt.figure(figsize=(10, 8))
    
    # Use different colors for different emotions
    emotion_colors = {
        "excitement": 'red',
        "happiness": 'orange',
        "contentment": 'yellow',
        "calmness": 'green',
        "neutral": 'gray',
        "sadness": 'blue',
        "fear": 'purple',
        "anger": 'darkred'
    }
    
    # Create color list for scatter plot
    colors = [emotion_colors.get(emotion, 'gray') for emotion in emotions]
    
    plt.scatter(valence, arousal, c=colors, s=100, alpha=0.7)
    
    # Add labels for each emotional quadrant
    plt.text(0.25, 0.9, "High Energy\nNegative", ha='center', fontsize=12)
    plt.text(0.75, 0.9, "High Energy\nPositive", ha='center', fontsize=12)
    plt.text(0.25, 0.1, "Low Energy\nNegative", ha='center', fontsize=12)
    plt.text(0.75, 0.1, "Low Energy\nPositive", ha='center', fontsize=12)
    
    # Create a legend
    for emotion, color in emotion_colors.items():
        if emotion in emotions:
            plt.scatter([], [], c=color, label=emotion, s=100)
    
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0.5, color='k', linestyle='--', alpha=0.3)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Valence (Negative to Positive)', fontsize=12)
    plt.ylabel('Arousal (Low to High)', fontsize=12)
    plt.title('Emotional Space: Valence vs. Arousal', fontsize=16)
    plt.legend(title="Emotions")
    
    plt.tight_layout()
    plt.savefig(dirs["audio"] / "valence_arousal.png", dpi=300)
    plt.close()

def save_analysis_data(features, emotion_data, dirs):
    print("Saving analysis data...")
    
    # Create a summary of the analysis
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
    
    # Save to JSON
    summary_path = dirs["audio"] / "visual_audio_analysis.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    
    # Create a text summary
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
    
    # Add peak moments for each emotion
    emotion_peaks = {}
    for segment in emotion_data["segments"]:
        emotion = segment["primary_emotion"]
        if emotion not in emotion_peaks:
            emotion_peaks[emotion] = []
        
        emotion_peaks[emotion].append((segment["arousal"] + segment["valence"])/2)
    
    # Find peak moments for each emotion
    text_summary.append("Peak Emotional Moments:")
    for emotion, intensities in emotion_peaks.items():
        if len(intensities) > 0:
            peak_idx = np.argmax(intensities)
            peak_segment = [s for s in emotion_data["segments"] if s["primary_emotion"] == emotion][peak_idx]
            text_summary.append(f"  - {emotion.capitalize()}: {peak_segment['start_time']:.1f}s - {peak_segment['end_time']:.1f}s")
    
    # Create a summary visualization
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
        
        # Setup directories
        dirs = setup_directories()
        
        # Path to the video
        video_path = dirs["temp"] / "ftrTyduMBFc.mp4"
        
        if not video_path.exists():
            print(f"Video not found at {video_path}")
            print("Please ensure the video exists at that location")
            sys.exit(1)
        
        # Extract visual features as proxies for audio
        features = extract_visual_audio_proxies(video_path)
        
        # Create visualizations
        create_proxy_waveform(features, dirs)
        create_feature_visualizations(features, dirs)
        create_energy_visualizations(features, dirs)
        create_heatmap_visualizations(features, dirs)
        
        # Map features to emotions
        emotion_data = map_features_to_emotion(features)
        
        # Create emotional visualizations
        create_emotional_visualizations(emotion_data, features, dirs)
        
        # Save analysis data
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