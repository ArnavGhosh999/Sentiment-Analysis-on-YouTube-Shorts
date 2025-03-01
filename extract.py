import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

def extract_all_comments(video_url, output_file="comments.csv"):
    """Extracts all YouTube comments using infinite scrolling until all comments are loaded."""

    # Setup Chrome options for independent execution
    options = Options()
    options.add_argument("--headless=new")  # Run in headless mode
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--incognito")  # Fresh session
    options.add_argument("--window-size=1920,1080")

    # Initialize WebDriver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(video_url)
    time.sleep(5)  # Allow page to load

    # Scroll until all comments are loaded
    body = driver.find_element(By.TAG_NAME, "body")
    last_height = driver.execute_script("return document.documentElement.scrollHeight")
    
    while True:
        body.send_keys(Keys.END)  # Scroll to the bottom
        time.sleep(2)

        new_height = driver.execute_script("return document.documentElement.scrollHeight")

        # If we can't scroll further, break
        if new_height == last_height:
            break
        last_height = new_height

    # Extract comments
    comments = []
    comment_elements = driver.find_elements(By.XPATH, '//*[@id="content-text"]')

    for i, comment in enumerate(comment_elements, start=1):
        comments.append([i, comment.text])

    # Close the browser
    driver.quit()

    # Save to CSV
    df = pd.DataFrame(comments, columns=["Serial Number", "Comment"])
    df.to_csv(output_file, index=False, encoding="utf-8-sig")  # utf-8-sig for emoji support

    print(f"✅ Extracted {len(comments)} comments and saved to {output_file}")

# Example Usage
video_url = "https://youtu.be/0L9yESn6JcQ"
extract_all_comments(video_url)
