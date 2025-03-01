from dotenv import load_dotenv
import os
import requests
import csv
import time

# Load environment variables
load_dotenv("config.env")
API_KEY = os.getenv("API_KEY")
VIDEO_ID = os.getenv("VIDEO_ID")

COMMENTS_FILE = "comments.csv"
DETAILS_FILE = "details.csv"

# Function to Fetch Video Details
def fetch_video_details():
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id={VIDEO_ID}&key={API_KEY}"
    response = requests.get(url).json()

    if "items" not in response or len(response["items"]) == 0:
        print("❌ Video not found!")
        return None

    video_info = response["items"][0]
    snippet = video_info["snippet"]
    stats = video_info["statistics"]

    details = {
        "Title": snippet["title"],
        "Channel": snippet["channelTitle"],
        "Published At": snippet["publishedAt"],
        "Views": stats.get("viewCount", "N/A"),
        "Likes": stats.get("likeCount", "N/A"),
        "Comments": stats.get("commentCount", "N/A"),
        "Channel ID": snippet["channelId"]
    }

    return details

# Function to Save Video Details to CSV
def save_video_details(details):
    with open(DETAILS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Title", "Channel", "Published At", "Views", "Likes", "Comments", "Channel ID"])
        writer.writerow(details.values())

# Function to Fetch All Comments
def fetch_all_comments():
    comments = []
    next_page_token = None

    print("⏳ Fetching comments...")

    while True:
        url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={VIDEO_ID}&key={API_KEY}&maxResults=100"
        if next_page_token:
            url += f"&pageToken={next_page_token}"

        response = requests.get(url).json()

        if "items" not in response:
            print("❌ No comments found!")
            break

        for item in response["items"]:
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append([
                snippet["authorDisplayName"],
                snippet["textDisplay"],
                snippet.get("likeCount", 0),
                snippet["publishedAt"]
            ])

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(1)  # Prevent API rate limiting

    return comments

# Function to Save Comments to CSV
def save_comments(comments):
    with open(COMMENTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Author", "Comment", "Likes", "Published At"])
        writer.writerows(comments)

# Main Function
def main():
    video_details = fetch_video_details()
    if video_details:
        save_video_details(video_details)
        print(f"✅ Video details saved in {DETAILS_FILE}")

    comments = fetch_all_comments()
    if comments:
        save_comments(comments)
        print(f"✅ {len(comments)} comments saved in {COMMENTS_FILE}")

if __name__ == "__main__":
    main()
