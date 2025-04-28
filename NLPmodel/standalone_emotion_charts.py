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
    
    viz_dirs = ["emotions", "timelines", "heatmaps", "transitions", "charts", "advanced_charts"]
    for viz_dir in viz_dirs:
        os.makedirs(audio_dir / viz_dir, exist_ok=True)
    
    return {
        "base": base_dir,
        "temp": temp_dir,
        "tone": tone_dir,
        "audio": audio_dir,
        "emotions": audio_dir / "emotions",
        "timelines": audio_dir / "timelines",
        "heatmaps": audio_dir / "heatmaps",
        "transitions": audio_dir / "transitions",
        "charts": audio_dir / "charts",
        "advanced": audio_dir / "advanced_charts"
    }

def analyze_video(video_path, sample_rate=5):
    print(f"Analyzing video: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    print(f"Video has {total_frames} frames at {fps} FPS, duration: {duration:.2f} seconds")
    
    sample_interval = max(1, int(fps / sample_rate))
    
    frame_data = []
    motion_data = []
    face_data = []
    
    prev_frame = None
    frame_count = 0
    time_points = []
    
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except:
        print("Warning: Could not load face detection model. Face detection will be skipped.")
        face_cascade = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % sample_interval == 0:
            time_point = frame_count / fps
            time_points.append(time_point)
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            brightness = np.mean(hsv[:,:,2])
            saturation = np.mean(hsv[:,:,1])
            hue_values = hsv[:,:,0]
            dominant_hue = np.bincount(hue_values.flatten()).argmax()
            
            face_count = 0
            if face_cascade is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                face_count = len(faces)
                
                face_data.append({
                    "time": time_point,
                    "face_count": face_count,
                    "faces": faces.tolist() if face_count > 0 else []
                })
            
            if prev_frame is not None:
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                
                diff = cv2.absdiff(prev_gray, curr_gray)
                motion_score = np.mean(diff) / 255.0
                
                h, w = diff.shape
                top_motion = np.mean(diff[:h//3, :]) / 255.0
                middle_motion = np.mean(diff[h//3:2*h//3, :]) / 255.0
                bottom_motion = np.mean(diff[2*h//3:, :]) / 255.0
                
                motion_data.append({
                    "time": time_point,
                    "motion_score": float(motion_score),
                    "top_motion": float(top_motion),
                    "middle_motion": float(middle_motion),
                    "bottom_motion": float(bottom_motion)
                })
            
            frame_data.append({
                "time": time_point,
                "brightness": float(brightness),
                "saturation": float(saturation),
                "dominant_hue": int(dominant_hue),
                "face_count": face_count
            })
            
            if time_point % 10 < (1/fps):
                sample_dir = Path(__file__).parent / "tone_analysis" / "audio" / "frames"
                os.makedirs(sample_dir, exist_ok=True)
                cv2.imwrite(str(sample_dir / f"frame_{time_point:.1f}s.jpg"), frame)
            
            prev_frame = frame
            
        frame_count += 1
        
        if frame_count % (sample_interval * 100) == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
    
    cap.release()
    
    if not motion_data:
        raise ValueError("No motion data extracted from video")
    
    return {
        "total_frames": total_frames,
        "fps": fps,
        "duration": duration,
        "time_points": np.array(time_points),
        "frame_data": frame_data,
        "motion_data": motion_data,
        "face_data": face_data
    }

def detect_emotions(video_data):
    print("Detecting emotions from visual features...")
    
    times = np.array([m["time"] for m in video_data["motion_data"]])
    motion = np.array([m["motion_score"] for m in video_data["motion_data"]])
    
    try:
        top_motion = np.array([m["top_motion"] for m in video_data["motion_data"]])
    except KeyError:
        top_motion = motion
    
    brightness = np.array([f["brightness"] for f in video_data["frame_data"]])
    saturation = np.array([f["saturation"] for f in video_data["frame_data"]])
    
    min_length = min(len(times), len(motion), len(brightness), len(saturation))
    times = times[:min_length]
    motion = motion[:min_length]
    top_motion = top_motion[:min_length]
    brightness = brightness[:min_length]
    saturation = saturation[:min_length]
    
    brightness_change = np.gradient(brightness)
    motion_change = np.gradient(motion)
    
    emotion_segments = []
    
    window_size = 5
    
    for i in range(0, len(times) - window_size + 1):
        window_start = i
        window_end = i + window_size
        time_start = times[window_start]
        time_end = times[window_end - 1]
        
        window_motion = np.mean(motion[window_start:window_end])
        window_brightness = np.mean(brightness[window_start:window_end])
        window_saturation = np.mean(saturation[window_start:window_end])
        window_motion_change = np.mean(np.abs(motion_change[window_start:window_end]))
        window_top_motion = np.mean(top_motion[window_start:window_end])
        
        norm_motion = min(window_motion / 0.2, 1.0)
        norm_brightness = min(window_brightness / 200, 1.0)
        norm_saturation = min(window_saturation / 200, 1.0)
        norm_motion_change = min(window_motion_change / 0.05, 1.0)
        norm_top_motion = min(window_top_motion / 0.2, 1.0)
        
        valence = norm_brightness * 0.6 + norm_saturation * 0.4
        arousal = norm_motion * 0.5 + norm_motion_change * 0.3 + norm_top_motion * 0.2
        tension = (1 - norm_brightness) * 0.3 + norm_motion_change * 0.5 + norm_motion * 0.2
        
        high_threshold = 0.65
        medium_threshold = 0.4
        low_threshold = 0.2
        
        if arousal > high_threshold:
            if valence > high_threshold:
                if tension > medium_threshold:
                    primary_emotion = "elation"
                else:
                    primary_emotion = "excitement"
            elif valence > medium_threshold:
                primary_emotion = "happiness"
            else:
                if tension > high_threshold:
                    primary_emotion = "anger"
                else:
                    primary_emotion = "frustration"
        elif arousal > medium_threshold:
            if valence > high_threshold:
                primary_emotion = "cheerfulness"
            elif valence > medium_threshold:
                if tension > medium_threshold:
                    primary_emotion = "anxiety"
                else:
                    primary_emotion = "contentment"
            else:
                primary_emotion = "distress"
        else:
            if valence > high_threshold:
                primary_emotion = "serenity"
            elif valence > medium_threshold:
                primary_emotion = "calmness"
            else:
                if tension > medium_threshold:
                    primary_emotion = "depression"
                else:
                    primary_emotion = "boredom"
        
        if norm_motion > 0.8 and np.abs(np.mean(brightness_change[window_start:window_end])) > 0.02:
            primary_emotion = "euphoria"
        
        if window_top_motion > 0.15 and window_motion > 0.1:
            primary_emotion = "crowd_excitement"
        
        emotion_segments.append({
            "time_start": float(time_start),
            "time_end": float(time_end),
            "valence": float(valence),
            "arousal": float(arousal),
            "tension": float(tension),
            "motion": float(norm_motion),
            "motion_change": float(norm_motion_change),
            "brightness": float(norm_brightness),
            "primary_emotion": primary_emotion,
            "intensity": float((arousal + valence) / 2)
        })
    
    emotion_counts = Counter([s["primary_emotion"] for s in emotion_segments])
    total_segments = len(emotion_segments)
    emotion_distribution = {e: c/total_segments for e, c in emotion_counts.items()}
    
    dominant_emotions = sorted(emotion_distribution.items(), key=lambda x: x[1], reverse=True)
    
    avg_valence = np.mean([s["valence"] for s in emotion_segments])
    avg_arousal = np.mean([s["arousal"] for s in emotion_segments])
    avg_tension = np.mean([s["tension"] for s in emotion_segments])
    
    transitions = []
    current_emotion = emotion_segments[0]["primary_emotion"] if emotion_segments else None
    
    for i, segment in enumerate(emotion_segments[1:], 1):
        if segment["primary_emotion"] != current_emotion:
            transitions.append({
                "time": segment["time_start"],
                "from_emotion": current_emotion,
                "to_emotion": segment["primary_emotion"]
            })
            current_emotion = segment["primary_emotion"]
    
    emotion_peaks = {}
    for emotion in emotion_distribution.keys():
        segments = [s for s in emotion_segments if s["primary_emotion"] == emotion]
        if segments:
            peak_segment = max(segments, key=lambda x: x["intensity"])
            emotion_peaks[emotion] = {
                "time_start": peak_segment["time_start"],
                "time_end": peak_segment["time_end"],
                "intensity": peak_segment["intensity"],
                "valence": peak_segment["valence"],
                "arousal": peak_segment["arousal"],
                "tension": peak_segment["tension"]
            }
    
    emotion_summary = {
        "dominant_emotions": dominant_emotions,
        "overall_dimensions": {
            "valence": float(avg_valence),
            "arousal": float(avg_arousal),
            "tension": float(avg_tension)
        },
        "emotion_distribution": emotion_distribution,
        "transition_count": len(transitions),
        "transitions": transitions,
        "segments": emotion_segments
    }
    
    return emotion_summary, emotion_peaks

def save_emotion_data(emotion_summary, emotion_peaks, dirs):
    print("Saving emotion data to files...")
    
    summary_path = dirs["audio"] / "emotion_summary.json"
    with open(summary_path, 'w') as f:
        summary_to_save = emotion_summary.copy()
        if "segments" in summary_to_save:
            del summary_to_save["segments"]
        json.dump(summary_to_save, f, indent=4, default=str)
    
    peaks_path = dirs["audio"] / "emotion_peaks.json"
    with open(peaks_path, 'w') as f:
        json.dump(emotion_peaks, f, indent=4, default=str)
    
    dominant_emotions = [f"{emotion} ({value:.1%})" for emotion, value in emotion_summary["dominant_emotions"][:3]]
    
    text_summary = [
        "EMOTION ANALYSIS SUMMARY",
        "------------------------",
        f"Overall Dominant Emotions: {', '.join(dominant_emotions)}",
        f"",
        f"Emotional Dimensions:",
        f"  - Valence (positivity): {emotion_summary['overall_dimensions']['valence']:.2f}",
        f"  - Arousal (energy): {emotion_summary['overall_dimensions']['arousal']:.2f}",
        f"  - Tension: {emotion_summary['overall_dimensions']['tension']:.2f}",
        f"",
        f"Emotion Transitions: {len(emotion_summary['transitions'])}",
        f"",
        f"Peak Emotional Moments:"
    ]
    
    for emotion, peak in emotion_peaks.items():
        text_summary.append(f"  - {emotion.capitalize()}: {peak['time_start']:.1f}s - {peak['time_end']:.1f}s (Intensity: {peak['intensity']:.2f})")
    
    plt.figure(figsize=(12, 8))
    plt.text(0.5, 0.5, "\n".join(text_summary), 
             horizontalalignment='center',
             fontsize=14, 
             linespacing=1.5)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(dirs["audio"] / "text_summary.png", dpi=300)
    plt.close()
    
    return summary_path, peaks_path

def create_basic_charts(emotion_summary, emotion_peaks, dirs):
    print("Creating basic emotion charts...")
    
    plt.figure(figsize=(12, 10))
    
    emotions = list(emotion_summary["emotion_distribution"].keys())
    values = list(emotion_summary["emotion_distribution"].values())
    
    sorted_data = sorted(zip(emotions, values), key=lambda x: x[1], reverse=True)
    emotions = [x[0] for x in sorted_data]
    values = [x[1] for x in sorted_data]
    
    emotion_colors = {
        "excitement": '#ff0000',
        "happiness": '#ff9500',
        "cheerfulness": '#ffcc00',
        "elation": '#ff00ff',
        "contentment": '#00cc00',
        "serenity": '#00ffcc',
        "calmness": '#00ccff',
        "boredom": '#cccccc',
        "anxiety": '#9900cc',
        "distress": '#990000',
        "frustration": '#cc6600',
        "anger": '#990099',
        "depression": '#000066',
        "euphoria": '#ff3399',
        "crowd_excitement": '#ffff00'
    }
    
    colors = [emotion_colors.get(e, '#cccccc') for e in emotions]
    
    wedges, texts, autotexts = plt.pie(
        values, 
        labels=emotions, 
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False,
        colors=colors,
        explode=[0.1 if i==0 else 0 for i in range(len(emotions))],
        shadow=True
    )
    
    for text in texts:
        text.set_fontsize(11)
    
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.title('Emotion Distribution', fontsize=16)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(dirs["charts"] / "emotion_distribution_pie.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 8))
    
    valence_values = []
    arousal_values = []
    emotions_list = []
    
    for emotion, data in emotion_peaks.items():
        valence_values.append(data["valence"])
        arousal_values.append(data["arousal"])
        emotions_list.append(emotion)
    
    scatter_colors = [emotion_colors.get(e, "#cccccc") for e in emotions_list]
    
    plt.scatter(valence_values, arousal_values, c=scatter_colors, s=200, alpha=0.7)
    
    for i, emotion in enumerate(emotions_list):
        plt.text(valence_values[i], arousal_values[i], emotion, 
                ha='center', va='center', fontsize=9, 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.text(0.25, 0.9, "High Energy\nNegative", ha='center', fontsize=12)
    plt.text(0.75, 0.9, "High Energy\nPositive", ha='center', fontsize=12)
    plt.text(0.25, 0.1, "Low Energy\nNegative", ha='center', fontsize=12)
    plt.text(0.75, 0.1, "Low Energy\nPositive", ha='center', fontsize=12)
    
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0.5, color='k', linestyle='--', alpha=0.3)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Valence (Negative to Positive)', fontsize=12)
    plt.ylabel('Arousal (Low to High)', fontsize=12)
    plt.title('Emotional Space: Valence vs. Arousal', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(dirs["charts"] / "valence_arousal.png", dpi=300)
    plt.close()
    
    segments = emotion_summary["segments"]
    times = [(s["time_start"] + s["time_end"]) / 2 for s in segments]
    emotions = [s["primary_emotion"] for s in segments]
    arousal = [s["arousal"] for s in segments]
    valence = [s["valence"] for s in segments]
    tension = [s["tension"] for s in segments]
    
    plt.figure(figsize=(16, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(times, arousal, 'r-', label='Arousal', linewidth=2)
    plt.plot(times, valence, 'g-', label='Valence', linewidth=2)
    plt.plot(times, tension, 'b-', label='Tension', linewidth=2)
    
    for transition in emotion_summary["transitions"]:
        plt.axvline(x=transition["time"], color='black', linestyle='--', alpha=0.5)
        
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Intensity', fontsize=12)
    plt.title('Emotional Dimensions Over Time', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    
    colors = [emotion_colors.get(e, '#cccccc') for e in emotions]
    
    plt.subplot(2, 1, 2)
    
    unique_emotions = set(emotions)
    
    for emotion in unique_emotions:
        emotion_times = [times[i] for i in range(len(times)) if emotions[i] == emotion]
        emotion_y = [0.5] * len(emotion_times)
        plt.scatter(emotion_times, emotion_y, label=emotion, 
                   c=[emotion_colors.get(emotion, '#cccccc')], s=100, alpha=0.7)
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.yticks([])
    plt.title('Emotional States Over Time', fontsize=16)
    plt.legend(title="Emotions", loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=5, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(dirs["charts"] / "emotion_timeline.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(14, 8))
    
    peak_times = [data["time_start"] for _, data in emotion_peaks.items()]
    peak_emotions = list(emotion_peaks.keys())
    peak_intensities = [data["intensity"] for _, data in emotion_peaks.items()]
    
    sort_indices = np.argsort(peak_intensities)[::-1]
    peak_times = [peak_times[i] for i in sort_indices]
    peak_emotions = [peak_emotions[i] for i in sort_indices]
    peak_intensities = [peak_intensities[i] for i in sort_indices]
    
    plt.barh(peak_emotions, peak_times, height=0.5, 
             color=[emotion_colors.get(e, "#cccccc") for e in peak_emotions])
    
    for i, (emotion, time, intensity) in enumerate(zip(peak_emotions, peak_times, peak_intensities)):
        plt.text(time + 1, i, f"{intensity:.2f}", va='center')
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.title('Key Emotional Moments', fontsize=16)
    plt.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(dirs["charts"] / "key_moments.png", dpi=300)
    plt.close()

def create_enhanced_charts(emotion_summary, emotion_peaks, dirs):
    print("Creating enhanced emotion charts...")
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    dominant_emotions = emotion_summary["dominant_emotions"][:6]
    emotions = [e[0] for e in dominant_emotions]
    values = [e[1] for e in dominant_emotions]
    
    values = [v * 100 for v in values]
    
    angles = np.linspace(0, 2*np.pi, len(emotions), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    emotions += emotions[:1]
    
    plt.plot(angles, values, 'o-', linewidth=2)
    plt.fill(angles, values, alpha=0.25)
    plt.xticks(angles[:-1], emotions[:-1], size=12)
    plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], 
              color="grey", size=10)
    
    for angle, value, emotion in zip(angles[:-1], values[:-1], emotions[:-1]):
        plt.text(angle, value + 5, f"{value:.1f}%", ha='center')
    
    plt.title("Dominant Emotions Radar Chart", size=16)
    
    plt.tight_layout()
    plt.savefig(dirs["advanced"] / "emotion_radar.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(12, len(emotion_peaks) // 2 + 3))
    
    emotions = []
    dimensions = []
    
    for emotion, data in emotion_peaks.items():
        emotions.append(emotion)
        dimensions.append([
            data["valence"],
            data["arousal"],
            data.get("tension", 0.5),
            data["intensity"]
        ])
    
    dim_array = np.array(dimensions)
    intensity_order = np.argsort([d[3] for d in dimensions])[::-1]
    
    emotions_sorted = [emotions[i] for i in intensity_order]
    dim_array_sorted = dim_array[intensity_order]
    
    sns.heatmap(
        dim_array_sorted,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=["Valence", "Arousal", "Tension", "Intensity"],
        yticklabels=emotions_sorted,
        linewidths=0.5
    )
    
    plt.title("Emotional Dimensions by Emotion", fontsize=16)
    plt.tight_layout()
    plt.savefig(dirs["advanced"] / "emotion_dimensions_heatmap.png", dpi=300)
    plt.close()
    
    if emotion_summary["transitions"]:
        try:
            import networkx as nx
            
            G = nx.DiGraph()
            
            top_emotions = set([e[0] for e in emotion_summary["dominant_emotions"][:8]])
            for emotion in top_emotions:
                G.add_node(emotion)
            
            transition_counts = {}
            for transition in emotion_summary["transitions"]:
                from_emotion = transition["from_emotion"]
                to_emotion = transition["to_emotion"]
                
                if from_emotion in top_emotions and to_emotion in top_emotions:
                    key = (from_emotion, to_emotion)
                    if key in transition_counts:
                        transition_counts[key] += 1
                    else:
                        transition_counts[key] = 1
            
            for (from_e, to_e), count in transition_counts.items():
                G.add_edge(from_e, to_e, weight=count)
            
            emotion_colors = {
                "excitement": '#ff0000',
                "happiness": '#ff9500',
                "cheerfulness": '#ffcc00',
                "elation": '#ff00ff',
                "contentment": '#00cc00',
                "serenity": '#00ffcc',
                "calmness": '#00ccff',
                "boredom": '#cccccc',
                "anxiety": '#9900cc',
                "distress": '#990000',
                "frustration": '#cc6600',
                "anger": '#990099',
                "depression": '#000066',
                "euphoria": '#ff3399',
                "crowd_excitement": '#ffff00'
            }
            
            plt.figure(figsize=(12, 10))
            
            pos = nx.spring_layout(G, seed=42)
            
            node_colors = [emotion_colors.get(n, "#cccccc") for n in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_size=1500, node_color=node_colors)
            
            edge_widths = [G[u][v]['weight'] for u, v in G.edges()]
            max_width = max(edge_widths) if edge_widths else 1
            norm_widths = [1 + (w / max_width) * 5 for w in edge_widths]
            
            nx.draw_networkx_edges(G, pos, width=norm_widths, edge_color='gray',
                                  arrows=True, arrowsize=15, arrowstyle='->')
            
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
            
            plt.title('Emotion Transitions Network', fontsize=16)
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(dirs["advanced"] / "emotion_transitions_network.png", dpi=300)
            plt.close()
        except ImportError:
            print("NetworkX not available, skipping transition network visualization")
    
    if emotion_summary["transitions"]:
        plt.figure(figsize=(16, 6))
        
        transitions = emotion_summary["transitions"]
        transitions.sort(key=lambda x: x["time"])
        
        timeline = []
        current_emotion = transitions[0]["from_emotion"]
        start_time = 0
        
        for transition in transitions:
            timeline.append({
                "emotion": current_emotion,
                "start": start_time,
                "end": transition["time"]
            })
            
            current_emotion = transition["to_emotion"]
            start_time = transition["time"]
        
        end_time = transitions[-1]["time"] + 10
        timeline.append({
            "emotion": current_emotion,
            "start": start_time,
            "end": end_time
        })
        
        emotion_colors = {
            "excitement": '#ff0000',
            "happiness": '#ff9500',
            "cheerfulness": '#ffcc00',
            "elation": '#ff00ff',
            "contentment": '#00cc00',
            "serenity": '#00ffcc',
            "calmness": '#00ccff',
            "boredom": '#cccccc',
            "anxiety": '#9900cc',
            "distress": '#990000',
            "frustration": '#cc6600',
            "anger": '#990099',
            "depression": '#000066',
            "euphoria": '#ff3399',
            "crowd_excitement": '#ffff00'
        }
        
        for segment in timeline:
            plt.plot(
                [segment["start"], segment["end"]], 
                [1, 1],
                linewidth=10, 
                solid_capstyle='butt',
                color=emotion_colors.get(segment["emotion"], "#cccccc")
            )
            mid_point = (segment["start"] + segment["end"]) / 2
            plt.text(mid_point, 1.1, segment["emotion"], 
                    ha='center', va='bottom', rotation=45, fontsize=9)
        
        transition_times = [t["time"] for t in transitions]
        plt.scatter(transition_times, [1] * len(transition_times), 
                   color='black', s=50, zorder=5)
        
        plt.yticks([])
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.title('Emotion Progression Timeline', fontsize=16)
        plt.grid(True, axis='x', alpha=0.3)
        
        plt.ylim(0.9, 1.3)
        
        plt.tight_layout()
        plt.savefig(dirs["advanced"] / "emotion_progression_timeline.png", dpi=300)
        plt.close()
    
    plt.figure(figsize=(14, 10))
    
    sorted_emotions = [e[0] for e in emotion_summary["dominant_emotions"]]
    
    y_positions = {}
    y_pos = 0
    
    for emotion in sorted_emotions:
        if emotion in emotion_peaks:
            peak = emotion_peaks[emotion]
            y_positions[emotion] = y_pos
            
            plt.scatter(peak["time_start"], y_pos, 
                       s=200, marker='*', 
                       c=emotion_colors.get(emotion, "#cccccc"))
            
            plt.text(peak["time_start"], y_pos + 0.2, 
                    f"{emotion}\n(Intensity: {peak['intensity']:.2f})", 
                    ha='center', fontsize=9)
            
            y_pos += 1
    
    plt.yticks(list(y_positions.values()), list(y_positions.keys()))
    
    plt.axhline(y=-0.5, xmin=0, xmax=1, color='black', linestyle='-', linewidth=2)
    
    for emotion, pos in y_positions.items():
        peak_time = emotion_peaks[emotion]["time_start"]
        plt.vlines(x=peak_time, ymin=-0.5, ymax=pos, 
                  linestyles='dashed', alpha=0.5)
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.title('Peak Emotional Moments Timeline', fontsize=16)
    plt.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(dirs["advanced"] / "peak_moments_timeline.png", dpi=300)
    plt.close()
    
    if emotion_summary["transitions"]:
        transitions = emotion_summary["transitions"]
        transitions.sort(key=lambda x: x["time"])
        
        all_emotions = set([t["from_emotion"] for t in transitions] + 
                          [t["to_emotion"] for t in transitions])
        
        timeline_points = []
        current_emotion = transitions[0]["from_emotion"]
        timeline_points.append((0, current_emotion))
        
        for transition in transitions:
            timeline_points.append((transition["time"] - 0.01, current_emotion))
            current_emotion = transition["to_emotion"]
            timeline_points.append((transition["time"], current_emotion))
        
        end_time = transitions[-1]["time"] + 10
        timeline_points.append((end_time, current_emotion))
        
        df = pd.DataFrame(timeline_points, columns=["time", "emotion"])
        
        pivot_data = []
        
        for time, emotion in timeline_points:
            row = {"time": time}
            for e in all_emotions:
                row[e] = 1 if e == emotion else 0
            pivot_data.append(row)
        
        df_pivot = pd.DataFrame(pivot_data)
        
        plt.figure(figsize=(14, 8))
        
        top_emotions = [e[0] for e in emotion_summary["dominant_emotions"]]
        emotions_to_plot = [e for e in top_emotions if e in all_emotions]
        
        x = df_pivot["time"]
        
        plt.stackplot(x, [df_pivot[e] for e in emotions_to_plot], 
                     labels=emotions_to_plot,
                     colors=[emotion_colors.get(e, "#cccccc") for e in emotions_to_plot],
                     alpha=0.8)
        
        plt.xlabel("Time (seconds)", fontsize=12)
        plt.ylabel("Emotion Presence", fontsize=12)
        plt.title("Emotions Over Time", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), 
                  ncol=min(5, len(emotions_to_plot)))
        
        plt.tight_layout()
        plt.savefig(dirs["advanced"] / "emotion_intensity_timeline.png", dpi=300)
        plt.close()
    
    if emotion_summary["transitions"]:
        transition_counts = {}
        
        for transition in emotion_summary["transitions"]:
            from_emotion = transition["from_emotion"]
            to_emotion = transition["to_emotion"]
            key = (from_emotion, to_emotion)
            
            if key in transition_counts:
                transition_counts[key] += 1
            else:
                transition_counts[key] = 1
        
        from_emotions = set(t["from_emotion"] for t in emotion_summary["transitions"])
        to_emotions = set(t["to_emotion"] for t in emotion_summary["transitions"])
        all_emotions = sorted(list(from_emotions.union(to_emotions)))
        
        matrix = np.zeros((len(all_emotions), len(all_emotions)))
        
        for i, from_emotion in enumerate(all_emotions):
            for j, to_emotion in enumerate(all_emotions):
                key = (from_emotion, to_emotion)
                if key in transition_counts:
                    matrix[i, j] = transition_counts[key]
        
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix_norm = matrix / row_sums
        
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(matrix_norm, annot=matrix.astype(int), fmt="d", 
                    xticklabels=all_emotions, yticklabels=all_emotions, 
                    cmap="YlGnBu", linewidths=0.5)
        
        plt.title("Emotion Transition Matrix", fontsize=16)
        plt.xlabel("To Emotion", fontsize=12)
        plt.ylabel("From Emotion", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(dirs["advanced"] / "emotion_transition_matrix.png", dpi=300)
        plt.close()
    
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        emotions = []
        valence_values = []
        arousal_values = []
        tension_values = []
        sizes = []
        
        for emotion, data in emotion_peaks.items():
            emotions.append(emotion)
            valence_values.append(data["valence"])
            arousal_values.append(data["arousal"])
            tension_values.append(data.get("tension", 0.5))
            
            dist_pct = emotion_summary["emotion_distribution"].get(emotion, 0.01)
            sizes.append(dist_pct * 5000)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = [emotion_colors.get(e, "#cccccc") for e in emotions]
        
        scatter = ax.scatter(valence_values, arousal_values, tension_values, 
                            s=sizes, c=colors, alpha=0.7)
        
        for i, emotion in enumerate(emotions):
            ax.text(valence_values[i], arousal_values[i], tension_values[i], 
                    emotion, fontsize=8)
        
        ax.set_xlabel('Valence', fontsize=12)
        ax.set_ylabel('Arousal', fontsize=12)
        ax.set_zlabel('Tension', fontsize=12)
        ax.set_title('3D Emotion Space', fontsize=16)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(dirs["advanced"] / "emotion_3d_plot.png", dpi=300)
        plt.close()
    except ImportError:
        print("3D plotting not available, skipping 3D emotion plot")

def main():
    try:
        print("\n*** COMPREHENSIVE EMOTION VISUALIZATION ***\n")
        
        dirs = setup_directories()
        
        video_path = dirs["temp"] / "ftrTyduMBFc.mp4"
        
        if not video_path.exists():
            print(f"Video not found at {video_path}")
            print("Please ensure the video exists at that location")
            sys.exit(1)
        
        print("Analyzing video...")
        video_data = analyze_video(video_path, sample_rate=10)
        
        print("Detecting emotions...")
        emotion_summary, emotion_peaks = detect_emotions(video_data)
        
        print("Saving emotion data...")
        summary_path, peaks_path = save_emotion_data(emotion_summary, emotion_peaks, dirs)
        
        print("Creating basic charts...")
        create_basic_charts(emotion_summary, emotion_peaks, dirs)
        
        print("Creating enhanced charts...")
        create_enhanced_charts(emotion_summary, emotion_peaks, dirs)
        
        print("\nEmotion analysis and visualization complete!")
        print(f"Results saved to: {dirs['audio']}")
        
        print("\nSummary of detected emotions:")
        for emotion, value in emotion_summary["dominant_emotions"][:5]:
            print(f"  - {emotion.capitalize()}: {value:.1%}")
        
        print("\nPeak emotional moments:")
        for emotion, peak in emotion_peaks.items():
            print(f"  - {emotion.capitalize()}: {peak['time_start']:.1f}s (Intensity: {peak['intensity']:.2f})")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()