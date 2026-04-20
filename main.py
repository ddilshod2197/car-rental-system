import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

def blog_post_analyzer(text):
    # So'zlar ro'yxatini oling
    words = word_tokenize(text)
    
    # Stop so'zlar ro'yxatidan tozalash
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]
    
    # Eng mashhur so'zlar ro'yxatini oling
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(10)
    
    # O'rtacha jumlalar uzunligini hisoblang
    sentences = sent_tokenize(text)
    avg_sentence_length = len(words) / len(sentences)
    
    # Sentiment (oddiy) aniqlang
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    
    return most_common_words, avg_sentence_length, sentiment

text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. 
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."""

most_common_words, avg_sentence_length, sentiment = blog_post_analyzer(text)

print("Eng mashhur so'zlar:")
for word, count in most_common_words:
    print(f"{word}: {count}")

print(f"\nO'rtacha jumlalar uzunligi: {avg_sentence_length}")

print(f"\nSentiment (oddiy): {sentiment}")
