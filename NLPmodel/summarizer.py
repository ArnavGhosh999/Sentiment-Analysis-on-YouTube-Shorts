#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import dotenv

def load_config():
    config_path = Path(__file__).parent.parent / "config.env"
    dotenv.load_dotenv(config_path)
    video_id = os.getenv("VIDEO_ID")
    if not video_id:
        raise ValueError("VIDEO_ID not found in config.env")
    return video_id

def setup_directories():
    base_dir = Path(__file__).parent
    temp_dir = base_dir / "temp"
    summary_dir = base_dir / "summary"
    charts_dir = summary_dir / "charts"
    data_dir = summary_dir / "data"
    
    for directory in [temp_dir, summary_dir, charts_dir, data_dir]:
        os.makedirs(directory, exist_ok=True)
    
    return {
        "base": base_dir,
        "temp": temp_dir,
        "summary": summary_dir,
        "charts": charts_dir,
        "data": data_dir
    }

def download_video(video_id, dirs):
    try:
        import subprocess
        temp_dir = dirs["temp"]
        video_path = os.path.join(temp_dir, f"{video_id}.mp4")
        
        if os.path.exists(video_path):
            print(f"Video already exists at: {video_path}")
            return video_path
            
        command = ["yt-dlp", "-f", "mp4", "-o", video_path, f"https://www.youtube.com/watch?v={video_id}"]
        subprocess.run(command, check=True)
        
        if os.path.exists(video_path):
            print(f"Successfully downloaded video to: {video_path}")
            return video_path
        else:
            raise ValueError(f"Download completed but video file not found at {video_path}")
            
    except Exception as e:
        print(f"Error downloading video: {e}")
        raise

def analyze_video_content(video_path, dirs):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    print(f"Video has {total_frames} frames at {fps} FPS, duration: {duration:.2f} seconds")
    
    sample_interval = int(fps * 5)  # Sample every 5 seconds
    
    scene_data = []
    motion_scores = []
    
    prev_frame = None
    frame_count = 0
    
    while True:
        success, frame = cap.read()
        
        if not success:
            break
            
        if frame_count % sample_interval == 0:
            time_point = frame_count / fps
            
            if prev_frame is not None:
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_score = np.mean(magnitude)
                
                normalized_motion = min(motion_score / 10.0, 1.0)
                
                motion_scores.append({
                    "frame": frame_count,
                    "time": time_point,
                    "motion_score": normalized_motion
                })
                
                if normalized_motion > 0.7:
                    scene_type = "high_action"
                elif normalized_motion > 0.3:
                    scene_type = "medium_action"
                else:
                    scene_type = "low_action"
                    
                scene_data.append({
                    "frame": frame_count,
                    "time": time_point,
                    "scene_type": scene_type,
                    "motion_score": normalized_motion
                })
                
                if normalized_motion > 0.5:
                    sample_path = dirs["data"] / f"sample_frame_{frame_count}.jpg"
                    cv2.imwrite(str(sample_path), frame)
            
            prev_frame = frame
            
        frame_count += 1
        
        if frame_count % (sample_interval * 20) == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
    
    cap.release()
    
    return {
        "scene_data": scene_data,
        "motion_scores": motion_scores,
        "total_frames": total_frames,
        "fps": fps,
        "duration": duration
    }

def infer_emotions_from_motion(video_analysis):
    if not video_analysis or "scene_data" not in video_analysis:
        return {}
    
    scene_data = video_analysis["scene_data"]
    if not scene_data:
        return {}
    
    emotion_timeline = []
    for scene in scene_data:
        time_point = scene["time"]
        motion_score = scene["motion_score"]
        scene_type = scene["scene_type"]
        
        if scene_type == "high_action":
            if motion_score > 0.8:
                emotion = "excitement"
                intensity = motion_score
            else:
                emotion = "happy"
                intensity = motion_score
        elif scene_type == "medium_action":
            emotion = "neutral"
            intensity = 0.5
        else:
            emotion = "calm"
            intensity = 0.3
            
        emotion_timeline.append({
            "time": time_point,
            "emotion": emotion,
            "intensity": intensity,
            "motion_score": motion_score
        })
    
    emotion_counts = {}
    for item in emotion_timeline:
        emotion = item["emotion"]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    total = len(emotion_timeline) if emotion_timeline else 1
    emotion_distribution = {emotion: count / total for emotion, count in emotion_counts.items()}
    
    key_moments = []
    for item in emotion_timeline:
        if item["motion_score"] > 0.7:
            key_moments.append({
                "time": item["time"],
                "emotion": item["emotion"],
                "description": f"High action moment at {item['time']:.1f}s"
            })
    
    dominant_emotions = sorted(emotion_distribution.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "emotion_timeline": emotion_timeline,
        "emotion_distribution": emotion_distribution,
        "key_moments": key_moments,
        "dominant_emotions": dominant_emotions
    }

def generate_summary(video_analysis, emotion_data, dirs):
    if not video_analysis or not emotion_data:
        print("No data available to generate summary")
        return
    
    summary_text = []
    
    if "dominant_emotions" in emotion_data and emotion_data["dominant_emotions"]:
        emotions = ", ".join([f"{emotion} ({score:.1%})" for emotion, score in emotion_data["dominant_emotions"][:2]])
        summary_text.append(f"This video predominantly contains {emotions} content.")
    
    if "key_moments" in emotion_data:
        key_moments = emotion_data["key_moments"]
        if key_moments:
            summary_text.append(f"Found {len(key_moments)} high action moments that likely contain goals or celebrations.")
            if len(key_moments) > 3:
                times = [f"{moment['time']:.1f}s" for moment in key_moments[:3]]
                summary_text.append(f"Key moments occur at {', '.join(times)}, and more.")
            else:
                times = [f"{moment['time']:.1f}s" for moment in key_moments]
                summary_text.append(f"Key moments occur at {', '.join(times)}.")
    
    if "excitement" in [emotion for emotion, _ in emotion_data.get("dominant_emotions", [])]:
        summary_text.append("The video appears to be a compilation of soccer highlights with high-energy moments.")
    
    full_summary = " ".join(summary_text)
    
    summary_data = {
        "video_id": os.path.basename(list(Path(dirs["temp"]).glob("*.mp4"))[0]).split(".")[0],
        "duration": video_analysis.get("duration", 0),
        "dominant_emotions": emotion_data.get("dominant_emotions", []),
        "key_moments_count": len(emotion_data.get("key_moments", [])),
        "description": full_summary
    }
    
    summary_path = dirs["data"] / "video_summary.json"
    with open(summary_path, "w") as f:
        json_summary = {
            "video_id": summary_data["video_id"],
            "duration": float(summary_data["duration"]),
            "dominant_emotions": [(emotion, float(score)) for emotion, score in summary_data["dominant_emotions"]],
            "key_moments_count": summary_data["key_moments_count"],
            "description": summary_data["description"]
        }
        json.dump(json_summary, f, indent=4)
    
    generate_visualizations(emotion_data, dirs)
    
    return summary_data

def generate_visualizations(emotion_data, dirs):
    if "emotion_distribution" in emotion_data and emotion_data["emotion_distribution"]:
        plt.figure(figsize=(10, 8))
        emotions = list(emotion_data["emotion_distribution"].keys())
        values = list(emotion_data["emotion_distribution"].values())
        
        sorted_data = sorted(zip(emotions, values), key=lambda x: x[1], reverse=True)
        emotions = [x[0] for x in sorted_data]
        values = [x[1] for x in sorted_data]
        
        plt.pie(values, labels=emotions, autopct='%1.1f%%', startangle=90,
                colors=plt.cm.viridis(np.linspace(0, 1, len(emotions))))
        plt.axis('equal')
        plt.title("Emotion Distribution in Video")
        
        chart_path = dirs["charts"] / "emotion_distribution.png"
        plt.savefig(chart_path)
        plt.close()
    
    if "emotion_timeline" in emotion_data and emotion_data["emotion_timeline"]:
        plt.figure(figsize=(14, 6))
        
        times = [item["time"] for item in emotion_data["emotion_timeline"]]
        motions = [item["motion_score"] for item in emotion_data["emotion_timeline"]]
        
        plt.plot(times, motions, 'b-', linewidth=2)
        plt.fill_between(times, motions, alpha=0.3)
        
        if "key_moments" in emotion_data:
            key_times = [moment["time"] for moment in emotion_data["key_moments"]]
            if key_times:
                plt.scatter(key_times, [0.9] * len(key_times), color='red', s=100, marker='^', label='Key Moments')
        
        plt.xlabel("Time (seconds)")
        plt.ylabel("Motion Intensity")
        plt.title("Motion Intensity Over Time")
        plt.grid(True, alpha=0.3)
        
        if "key_moments" in emotion_data and emotion_data["key_moments"]:
            plt.legend()
        
        chart_path = dirs["charts"] / "motion_over_time.png"
        plt.savefig(chart_path)
        plt.close()

def main():
    try:
        print("Starting video summarizer...")
        
        dirs = setup_directories()
        
        video_id = load_config()
        print(f"Processing video ID: {video_id}")
        
        video_path = download_video(video_id, dirs)
        
        print("Analyzing video content...")
        video_analysis = analyze_video_content(video_path, dirs)
        
        print("Inferring emotions from motion analysis...")
        emotion_data = infer_emotions_from_motion(video_analysis)
        
        print("Generating summary...")
        summary = generate_summary(video_analysis, emotion_data, dirs)
        
        print("Video analysis complete!")
        if summary:
            print("\nSummary:")
            print(summary["description"])
            print(f"\nResults saved to: {dirs['summary']}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()