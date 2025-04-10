import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK’s sentiment tool (run once)
nltk.download('vader_lexicon')

# Load news data from Person B’s file
news_data = pd.read_csv('news.csv')
print("News data loaded:")
print(news_data)

# Set up the summarizer with longer output
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
def summarize_text(text):
    try:
        # Adjusted max_length for 4-5 sentences (around 100-150 tokens)
        summary = summarizer(text, max_length=150, min_length=80, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Summary failed for text: {text[:50]}...: {e}")
        return "Summary failed"
news_data['summary'] = news_data['text'].apply(summarize_text)
print("\nNews with summaries:")
print(news_data[['title', 'summary']])

# Set up the matcher for query answering
matcher = SentenceTransformer('all-MiniLM-L6-v2')
query = "Why is the Invesco QQQ ETF down today?"
try:
    news_vectors = matcher.encode(news_data['summary'].tolist(), convert_to_tensor=True)
    query_vector = matcher.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_vector, news_vectors)
    best_match_index = scores.argmax().item()
    best_news = news_data.iloc[best_match_index]
    print(f"\nQuery: {query}")
    print(f"Best matching news: {best_news['title']}")
    print(f"Summary: {best_news['summary']}")
except Exception as e:
    print(f"Semantic search failed: {e}")

# Set up sentiment checker
sentiment_analyzer = SentimentIntensityAnalyzer()
def get_sentiment(text):
    try:
        scores = sentiment_analyzer.polarity_scores(text)
        return scores['compound']
    except:
        return 0.0
news_data['sentiment'] = news_data['text'].apply(get_sentiment)

# Load fund data from Person A’s file
fund_data = pd.read_csv('funds.csv')
try:
    drop_date = fund_data['date'][0]  # Using first date for simplicity
    negative_news = news_data[(news_data['sentiment'] < 0) & (news_data['date'] == drop_date)]
    print(f"\nNegative news on {drop_date} possibly causing QQQ drop:")
    print(negative_news[['title', 'summary', 'sentiment']])
except Exception as e:
    print(f"Fund matching failed: {e}")

# Save for Person D
news_data.to_csv('processed_news.csv', index=False)
print("\nSaved to 'processed_news.csv' for Person D!")




# below code which gives only 2 sentnce of output, but the above one gives 4 to 5 sentence.



# import pandas as pd
# from transformers import pipeline
# from sentence_transformers import SentenceTransformer, util
# import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer

# # Download NLTK’s sentiment tool (run once)
# nltk.download('vader_lexicon')

# # Load news data from Person B’s file
# news_data = pd.read_csv('news.csv')
# print("News data loaded:")
# print(news_data)

# # Set up the summarizer with adjusted lengths for short inputs
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# def summarize_text(text):
#     try:
#         # Adjusted for short inputs: aim for 2-3 sentences
#         summary = summarizer(text, max_length=40, min_length=20, do_sample=False)
#         return summary[0]['summary_text']
#     except Exception as e:
#         print(f"Summary failed for text: {text[:50]}...: {e}")
#         return "Summary failed"
# news_data['summary'] = news_data['text'].apply(summarize_text)
# print("\nNews with summaries:")
# print(news_data[['title', 'summary']])

# # Set up the matcher for query answering
# matcher = SentenceTransformer('all-MiniLM-L6-v2')
# query = "Why is the Invesco QQQ ETF down today?"
# try:
#     news_vectors = matcher.encode(news_data['summary'].tolist(), convert_to_tensor=True)
#     query_vector = matcher.encode(query, convert_to_tensor=True)
#     scores = util.cos_sim(query_vector, news_vectors)
#     best_match_index = scores.argmax().item()
#     best_news = news_data.iloc[best_match_index]
#     print(f"\nQuery: {query}")
#     print(f"Best matching news: {best_news['title']}")
#     print(f"Summary: {best_news['summary']}")
# except Exception as e:
#     print(f"Semantic search failed: {e}")

# # Set up sentiment checker
# sentiment_analyzer = SentimentIntensityAnalyzer()
# def get_sentiment(text):
#     try:
#         scores = sentiment_analyzer.polarity_scores(text)
#         return scores['compound']
#     except:
#         return 0.0
# news_data['sentiment'] = news_data['text'].apply(get_sentiment)

# # Load fund data from Person A’s file
# fund_data = pd.read_csv('funds.csv')
# try:
#     drop_date = fund_data['date'][0]  # Using first date for simplicity
#     negative_news = news_data[(news_data['sentiment'] < 0) & (news_data['date'] == drop_date)]
#     print(f"\nNegative news on {drop_date} possibly causing QQQ drop:")
#     print(negative_news[['title', 'summary', 'sentiment']])
# except Exception as e:
#     print(f"Fund matching failed: {e}")

# # Save for Person D
# news_data.to_csv('processed_news.csv', index=False)
# print("\nSaved to 'processed_news.csv' for Person D!")