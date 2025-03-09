import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_file = "sentiment_analysis.csv"
df = pd.read_csv(input_file)

df['Published At'] = pd.to_datetime(df['Published At'], errors='coerce')
df = df.dropna(subset=['Published At'])  
daily_sentiment = df.groupby(['Published At', 'Emotion']).size().reset_index(name='Count')
most_common_sentiment = daily_sentiment.loc[daily_sentiment.groupby('Published At')['Count'].idxmax()]

plt.figure(figsize=(12, 6))
sns.barplot(x=most_common_sentiment['Published At'].dt.strftime('%Y-%m-%d'), 
            y=most_common_sentiment['Count'], 
            hue=most_common_sentiment['Emotion'], 
            palette='pastel')
plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("Count")
plt.title("Most Common Sentiment by Day")
plt.legend(title="Emotion")
plt.tight_layout()
plt.show()
