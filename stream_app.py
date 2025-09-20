import streamlit as st
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Data Loading and Preprocessing ---

# Ensure NLTK data is downloaded
try:
    # This will check if 'vader_lexicon' is available and download if not.
    # We do this in a try-except block to handle cases where it's already downloaded.
    SentimentIntensityAnalyzer()
except LookupError:
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

# Initialize the SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

def preprocess_text(text):
    """
    Cleans and preprocesses the text for sentiment analysis.
    This includes converting to lowercase, removing special characters,
    and removing stopwords.
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, numbers, and links
    text = re.sub(r'http\S+|www.\S+|\S+\.com\S+|[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize words
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    # Join back into a string
    return ' '.join(filtered_words)

# --- Streamlit App UI and Logic ---

st.set_page_config(page_title="Streamlit Sentiment Analyzer", layout="centered")

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main-header {
        text-align: center;
        font-size: 2.5em;
        color: #1a1a1a;
        margin-bottom: 0.5em;
    }
    .text-area-label {
        font-size: 1.2em;
        font-weight: 500;
    }
    .stButton>button {
        width: 100%;
        border-radius: 25px;
        padding: 10px 0;
        font-size: 1.1em;
        font-weight: bold;
        color: white;
        background-color: #4CAF50;
        border: none;
        transition: transform 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        background-color: #45a049;
    }
    .results-box {
        margin-top: 2em;
        padding: 1.5em;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .positive {
        background-color: #e6ffed;
        color: #1f782c;
        border: 2px solid #1f782c;
    }
    .negative {
        background-color: #ffeded;
        color: #b33939;
        border: 2px solid #b33939;
    }
    .neutral {
        background-color: #f0f3ff;
        color: #2e59a8;
        border: 2px solid #2e59a8;
    }
    .sentiment-title {
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .score-list {
        font-size: 1em;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

st.title("Sentiment Analysis with Streamlit")
st.markdown("Enter some text below to get the sentiment analysis results.")

# Text input area
user_input = st.text_area("Enter your text here:", height=150, help="Type or paste any text you want to analyze.")

# Predict button
if st.button("Analyze"):
    if user_input:
        # Preprocess the input text
        processed_text = preprocess_text(user_input)
        
        # Get sentiment scores
        sentiment_scores = analyzer.polarity_scores(processed_text)
        
        # Determine the overall sentiment
        compound_score = sentiment_scores['compound']
        if compound_score >= 0.05:
            sentiment = "Positive"
            sentiment_color = "positive"
        elif compound_score <= -0.05:
            sentiment = "Negative"
            sentiment_color = "negative"
        else:
            sentiment = "Neutral"
            sentiment_color = "neutral"
        
        # Display results in a styled box
        st.markdown(f"""
        <div class="results-box {sentiment_color}">
            <h3 class="sentiment-title">Overall Sentiment: {sentiment}</h3>
            <div class="score-list">
                <ul>
                    <li><strong>Positive Score:</strong> {sentiment_scores['pos']:.2f}</li>
                    <li><strong>Negative Score:</strong> {sentiment_scores['neg']:.2f}</li>
                    <li><strong>Neutral Score:</strong> {sentiment_scores['neu']:.2f}</li>
                    <li><strong>Compound Score:</strong> {sentiment_scores['compound']:.2f}</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.warning("Please enter some text to analyze.")
