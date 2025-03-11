<h1 align="center"> Sentiment Analysis on YouTube Shorts NIT KURUKSHETRA project </h1>

<p align = "justify"> YouTube does not directly provide content creators with a built-in sentiment analysis graph showing viewers' sentiment towards their videos on the platform.However, understanding audience sentiment is crucial for creators to improve their content, engage with their community, and tailor their videos based on viewer feedback. By analyzing viewer comments, we aim to offer creators a comprehensive sentiment report, helping them gain deeper insights into audience reactions and trends over time. This feedback mechanism will empower creators to make data-driven decisions, refine their content strategies, and enhance viewer satisfaction. </p>

Key points :

- Scrapping data using the YouTube Data API and storing it in `comments.csv` file
- Using the `comments.csv` to train the SENTIMENT ANALYSIS model.
- Using the `VADER (Valence Aware Dictionary and Sentiment Reasoner)` for getting the Sentiment Analysis report.

<h2>📁 File structure</h2>
<pre>
PROJECT_FILE
│── .env                      # Environment variables (API keys, credentials)
│── Scrapper.py               # Script for scraping YouTube comments
│── comments.csv              # Scraped YouTube comments dataset
│── details.csv               # Processed data or additional details
│── sentiment.ipynb           # Jupyter Notebook for sentiment analysis
│── sentiment_analysis.csv    # Output file with sentiment analysis results
│── each_day                  # Extracting highest sentiment of each day
│── config.env                # Configuration settings
</pre>

<p align="center"><img src = 'IMAGES/VADER.png' height = 400, width =600></p>

- General representation of how VADER works.

<p align = "justify">VADER (Valence Aware Dictionary and Sentiment Reasoner) is a rule-based sentiment analysis tool designed for social media and short text. It assigns sentiment scores to words, considering intensity and context, including punctuation, capitalization, and emojis.</p>
  
<img src = 'IMAGES/output.png' height = 400, width =700>

- The Average Sentiment Analysis Score graph for a demo video given for training.

```
plt.figure(figsize=(10,5))
sns.histplot(df["Sentiment_Score"], bins=20, kde=True, color='blue')
plt.axvline(average_sentiment, color='red', linestyle='dashed', label=f'Average Score: {average_sentiment:.4f}')
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.title("Distribution of Sentiment Scores")
plt.legend()
plt.show()
```
   
<img src = 'IMAGES/each_day.py.jpg'>

<p align = "justify">Printed out what the highest sentiment from each day looks like(above)</p>