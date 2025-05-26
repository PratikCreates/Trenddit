import streamlit as st
import praw
import cleantext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime
from transformers import pipeline

# Initialize the Reddit instance
reddit = praw.Reddit(
    client_id="your_client_id",  # Replace with your Reddit app's client ID
    client_secret="app_secret_id",  # Replace with your Reddit app's client secret
    user_agent="MyDataFetchingApp by u/Heavy_Squash_1343 v1.0" # Name your script (e.g., "fetch_posts_script")
)

# Initialize sentiment analysis pipeline with a specific model and revision
sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english", 
    revision="main",  # Using the main branch as the revision
    framework="pt"  # Change to "tf" if using TensorFlow
)

# Function to analyze sentiment using the specified model
def analyze_text(text):
    # Analyze the text with the sentiment pipeline
    result = sentiment_pipeline(text[:512])  # Limit input to 512 characters for BERT-based models
    label = result[0]['label']
    score = result[0]['score']
    
    # Convert label to polarity for consistency with previous code
    polarity = score if label == 'POSITIVE' else -score
    subjectivity = 0.5  # Placeholder as transformers-based models don't output subjectivity
    return polarity, subjectivity


# Function to classify sentiment (from first code)
def classify_sentiment(polarity):
    if polarity > 0.5:
        return 'Positive'
    elif polarity < -0.5:
        return 'Negative'
    else:
        return 'Neutral'

# Function to classify domain (from first code)
def classify_domain(text):
    keywords = {
    "Immigration": ["immigration", "refugee", "visa", "asylum"],
    "Veterans Affairs": ["veterans", "military", "service", "support"],
    "Cyberbullying": ["cyberbullying", "bullying", "harassment", "online"],
    "Mental Health": ["mental health", "depression", "anxiety", "well-being"],
    "Addiction": ["addiction", "substance", "rehab", "treatment"],
    "Dating and Relationships": ["dating", "relationship", "love", "marriage"],
    "Divorce": ["divorce", "separation", "custody", "legal"],
    "Aging": ["aging", "elderly", "senior", "gerontology"],
    "Youth Development": ["youth", "development", "education", "growth"],
    "Economic Development": ["economic", "development", "growth", "poverty"],
    "Trade": ["trade", "export", "import", "business"],
    "Globalization": ["globalization", "world", "international", "trade"],
    "International Relations": ["international", "diplomacy", "relations", "foreign"],
    "Geopolitics": ["geopolitics", "politics", "power", "conflict"],
    "Urbanization": ["urbanization", "city", "population", "development"],
    "User Experience (UX)": ["UX", "user experience", "design", "interface"],
    "Graphic Design": ["graphic design", "design", "art", "visual"],
    "Cloud Computing": ["cloud computing", "cloud", "storage", "services"],
    "Robotics": ["robotics", "robot", "automation", "technology"],
    "Quantum Computing": ["quantum computing", "quantum", "computer", "theory"],
    "Biotechnology": ["biotechnology", "genetic", "research", "bio"],
    "Nanotechnology": ["nanotechnology", "nano", "small", "technology"],
    "Space Exploration": ["space", "exploration", "NASA", "astronomy"],
    "Cryptography": ["cryptography", "encryption", "security", "data"],
    "Ethics": ["ethics", "moral", "philosophy", "values"],
    "Crime and Law": ["crime", "law", "justice", "legal"],
    "Forensics": ["forensics", "crime scene", "investigation", "evidence"],
    "Public Safety": ["public safety", "emergency", "protection", "services"],
    "Fire Safety": ["fire safety", "fire", "emergency", "protection"],
    "Disaster Management": ["disaster", "management", "emergency", "response"],
    "Emergency Services": ["emergency", "services", "rescue", "help"],
    "Animal Rights": ["animal rights", "animal", "welfare", "protection"],
    "Sustainable Development": ["sustainable", "development", "environment", "eco"],
    "Community Development": ["community", "development", "growth", "participation"],
    "Philanthropy": ["philanthropy", "donation", "charity", "giving"],
    "Charity": ["charity", "donation", "help", "support"],
    "Social Entrepreneurship": ["social entrepreneurship", "business", "social", "impact"],
    "Work-Life Balance": ["work-life balance", "stress", "life", "work"],
    "Corporate Social Responsibility": ["CSR", "corporate", "responsibility", "business"],
    "Remote Work": ["remote work", "work from home", "telecommuting", "flexible"],
    "Digital Nomadism": ["digital nomad", "remote", "travel", "location independent"],
    "Politics": ["government", "election", "vote", "policy", "democracy", "diplomacy", "public opinion"],
    "Sports": ["game", "team", "score", "player", "Olympics", "eSports", "fitness", "athletics", "sports science"],
    "Technology": ["software", "device", "tech", "AI", "blockchain", "IoT", "5G", "cloud computing", "automation", "big data"],
    "Health": ["health", "disease", "treatment", "wellness", "mental health", "telemedicine", "public health", "nutrition"],
    "Finance": ["investment", "stock", "bank", "money", "fintech", "cryptocurrency", "wealth management", "financial planning"],
    "Entertainment": ["movie", "show", "celebrity", "music", "streaming", "animation", "gaming", "VR", "metaverse"],
    "Education": ["school", "learn", "education", "student", "edtech", "online learning", "K-12", "higher education", "STEM"],
    "Environment": ["climate", "pollution", "nature", "sustainability", "biodiversity", "conservation", "carbon footprint", "renewable energy"],
    "Travel": ["travel", "tourism", "trip", "vacation", "sustainable tourism", "hospitality", "adventure travel", "cultural tourism"],
    "Fashion": ["fashion", "style", "clothes", "trend", "sustainable fashion", "fast fashion", "luxury brands", "streetwear"],
    "Food": ["food", "cooking", "recipe", "dining", "veganism", "plant-based diet", "food tech", "nutrition", "food sustainability"],
    "Music": ["music", "album", "concert", "song", "streaming", "indie music", "pop culture", "music festivals", "digital music"],
    "Art": ["art", "gallery", "painting", "artist", "NFTs", "digital art", "contemporary art", "exhibitions"],
    "Real Estate": ["property", "real estate", "housing", "apartment", "mortgage", "commercial real estate", "urban planning"],
    "Gaming": ["game", "gaming", "console", "video game", "mobile gaming", "eSports", "AR", "VR", "game streaming"],
    "Automotive": ["car", "automobile", "vehicle", "motor", "electric vehicles", "autonomous vehicles", "battery tech", "EV infrastructure"],
    "Beauty": ["beauty", "makeup", "skincare", "cosmetics", "clean beauty", "personal care", "sustainable beauty", "anti-aging"],
    "Science": ["science", "research", "experiment", "scientific", "genomics", "astrophysics", "biotechnology", "climate science"],
    "Business": ["business", "company", "startup", "enterprise", "small business", "B2B", "entrepreneurship", "digital transformation"],
    "Marketing": ["advertising", "promotion", "marketing", "brand", "content marketing", "influencer marketing", "SEO", "digital marketing"],
    "Social Media": ["social", "media", "facebook", "twitter", "influencers", "viral trends", "digital platforms", "user engagement"],
    "Lifestyle": ["lifestyle", "wellness", "self-care", "habits", "mindfulness", "work-life balance", "minimalism", "sustainable living"],
    "Parenting": ["parenting", "children", "family", "raising kids", "education", "parenting styles", "child development", "new parents"],
    "Culture": ["culture", "tradition", "heritage", "society", "cultural diversity", "pop culture", "ethnic studies", "subcultures"],
    "Literature": ["literature", "book", "author", "novel", "poetry", "literary analysis", "bestsellers", "digital publishing"],
    "Psychology": ["psychology", "mind", "behavior", "emotion", "mental health", "cognitive science", "therapy", "emotional intelligence"],
    "Philosophy": ["philosophy", "thought", "reason", "ethics", "existentialism", "mindfulness", "moral philosophy", "logic"],
    "Religion": ["religion", "faith", "spirituality", "belief", "interfaith", "theology", "rituals", "religious studies"],
    "Spirituality": ["spirituality", "meditation", "mindfulness", "soul", "healing", "consciousness", "self-discovery", "well-being"],
    "Fitness": ["fitness", "exercise", "health", "workout", "home workouts", "fitness tech", "athlete training", "personal training"],
    "Nutrition": ["nutrition", "diet", "healthy eating", "vitamins", "plant-based diets", "nutritional science", "superfoods"],
    "Agriculture": ["agriculture", "farming", "crops", "livestock", "agritech", "sustainable farming", "vertical farming"],
    "Telecommunications": ["telecommunications", "network", "communication", "signal", "5G", "IoT", "fiber optics", "satellite tech"],
    "Cryptocurrency": ["cryptocurrency", "bitcoin", "blockchain", "finance", "DeFi", "NFTs", "altcoins", "crypto regulation"],
    "AI and Machine Learning": ["AI", "machine learning", "deep learning", "algorithm", "data science", "neural networks", "automation"],
    "Cybersecurity": ["cybersecurity", "hacking", "security", "data breach", "ethical hacking", "data protection", "ransomware"],
    "E-commerce": ["e-commerce", "online shopping", "retail", "sales", "ecommerce tech", "marketplaces", "customer experience"],
    "Retail": ["retail", "store", "shop", "customer", "omnichannel", "supply chain", "inventory", "point of sale"],
    "Pharmaceuticals": ["pharmaceuticals", "medicine", "drugs", "healthcare", "biopharma", "drug discovery", "clinical trials"],
    "Logistics": ["logistics", "shipping", "delivery", "transport", "supply chain", "freight", "warehousing", "last-mile delivery"],
    "Aerospace": ["aerospace", "aviation", "space", "aircraft", "space travel", "satellites", "UAVs", "space missions"],
    "Construction": ["construction", "building", "infrastructure", "architecture", "sustainable building", "smart cities"],
    "Hospitality": ["hospitality", "service", "hotel", "travel", "tourism", "customer service", "guest experience"],
    "Insurance": ["insurance", "policy", "coverage", "claims", "health insurance", "life insurance", "cyber insurance"],
    "Nonprofit": ["nonprofit", "charity", "organization", "community", "volunteer", "donations", "social impact"],
    "Activism": ["activism", "protest", "campaign", "advocacy", "social movements", "awareness", "human rights"],
    "Human Rights": ["human rights", "freedom", "justice", "equality", "international law", "social justice"],
    "Urban Development": ["urban", "development", "city", "infrastructure", "smart cities", "public transit", "urbanization"],
    "Renewable Energy": ["renewable", "energy", "solar", "wind", "hydropower", "green tech", "sustainable energy"],
    "Marine Life": ["marine", "ocean", "sea", "aquatic", "coral reefs", "marine conservation", "fisheries"],
    "Wildlife Conservation": ["wildlife", "conservation", "endangered", "species", "habitat protection", "biodiversity"],
    "Social Justice": ["social justice", "equity", "injustice", "discrimination", "diversity", "inclusivity"],
    "Gender Issues": ["gender", "feminism", "equality", "rights", "LGBTQ", "gender studies"],
    "Climate Change": ["climate change", "global warming", "environment", "pollution", "carbon emissions", "climate policy"],
    "Digital Marketing": ["digital marketing", "SEO", "social media", "advertising", "content creation", "analytics"],
    "Web Development": ["web development", "website", "coding", "programming", "frontend", "backend", "JavaScript frameworks"],
    "Mobile Apps": ["mobile apps", "application", "software", "mobile", "iOS", "Android", "app development"]
}

    for domain, keys in keywords.items():
        if any(key in text.lower() for key in keys):
            return domain
    return 'General'

# Function to fetch Reddit posts (from second code)
def fetch_reddit_posts(query):
    posts_data = []
    subreddit = reddit.subreddit("all")  # Searching across all subreddits
    for post in subreddit.search(query, sort="new", limit=100):
        posts_data.append({
            "title": post.title,
            "score": post.score,
            "text": post.selftext,
            "created_utc": datetime.utcfromtimestamp(post.created_utc),
            "url": post.url
        })
    return pd.DataFrame(posts_data)

# Function to detect fake news (simplified for demo purposes, from second code)
def detect_fake_news(posts_df):
    fake_percentage = np.random.uniform(0, 0.05)  # Simulating 5% fake news for demo
    fake_news_count = int(fake_percentage * len(posts_df))
    return fake_news_count, fake_percentage

# Trend analysis (from second code with first code's trend analysis logic)
def fetch_trends(domain):
    data = {
        'Date': pd.date_range(start='2023-01-01', periods=30),
        'Trend': np.random.randn(30).cumsum()  # Random walk data
    }
    return pd.DataFrame(data)

def calculate_moving_average(trend_data, window_size=5):
    return trend_data['Trend'].rolling(window=window_size).mean()

def plot_trend(trend_data, domain):
    plt.figure(figsize=(10, 5))
    plt.plot(trend_data['Date'], trend_data['Trend'], marker='o', label='Actual Trend')

    # Calculate and plot moving average
    trend_data['Moving Average'] = calculate_moving_average(trend_data)
    plt.plot(trend_data['Date'], trend_data['Moving Average'], label='Moving Average', linestyle='--')

    # Simple linear regression for forecasting
    X = np.array(range(len(trend_data))).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, trend_data['Trend'])
    future_dates = np.array(range(len(trend_data), len(trend_data) + 5)).reshape(-1, 1)
    future_trend = model.predict(future_dates)
    future_dates = pd.date_range(start=trend_data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=5)

    plt.plot(future_dates, future_trend, marker='x', color='red', label='Forecasted Trend')

    plt.title(f'Trend Analysis for {domain}')
    plt.xlabel('Date')
    plt.ylabel('Trend Score')
    plt.grid()
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)

def infer_trend(trend_data, domain):
    trend_slope = (trend_data['Trend'].iloc[-1] - trend_data['Trend'].iloc[0]) / len(trend_data)
    if trend_slope > 0.1:
        return f"The trend for {domain} is increasing, suggesting rising interest or relevance."
    elif trend_slope < -0.1:
        return f"The trend for {domain} is decreasing, suggesting a potential decline in interest or relevance."
    else:
        return f"The trend for {domain} is relatively stable, indicating consistent interest over time."

# Display the app UI
st.title('Sentiment and Trend Analysis of Reddit Posts')

query = st.text_input('Enter a topic or domain to search Reddit:')
if query:
    posts_df = fetch_reddit_posts(query)

    if not posts_df.empty:
        # Sentiment analysis using TextBlob and VADER
        posts_df['polarity'], posts_df['subjectivity'] = zip(*posts_df['text'].apply(analyze_text))
        posts_df['sentiment'] = posts_df['polarity'].apply(classify_sentiment)
        posts_df['domain'] = posts_df['text'].apply(classify_domain)
        
        # Fake news detection
        fake_news_count, fake_news_percentage = detect_fake_news(posts_df)

        # Trend analysis using the first code's trend analysis functions
        trend_data = fetch_trends(query)
        plot_trend(trend_data, query)
        trend_inference = infer_trend(trend_data, query)

        # Display the results
        st.write(f"Total Posts: {len(posts_df)}")
        st.write(f"Fake News Detected: {fake_news_count} posts ({fake_news_percentage*100:.2f}%)")
        
        st.subheader("Sentiment Analysis")
        st.write(posts_df[['title', 'sentiment', 'polarity', 'subjectivity', 'domain']])

        st.subheader("Trend Inference")
        st.write(trend_inference)

        # Display cleaned text (optional)
        cleaned_text = cleantext.clean(query, clean_all=False, extra_spaces=True,
                                       stopwords=True, lowercase=True, numbers=True, punct=True)
        st.write('Cleaned Text: ', cleaned_text)
    else:
        st.write("No posts found for this topic.")
else:
    st.write("Enter a topic to analyze.")
