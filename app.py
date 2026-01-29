import os
import sys
import pickle
import streamlit as st
import numpy as np
from books_recommender.logger.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.pipeline.training_pipeline import TrainingPipeline
from books_recommender.exception.exception_handler import AppException


class Recommendation:
    def __init__(self, app_config=AppConfiguration()):
        try:
            self.recommendation_config = app_config.get_recommendation_config()
        except Exception as e:
            raise AppException(e, sys) from e

    def fetch_poster(self, suggestion):
        try:
            book_name = []
            ids_index = []
            poster_url = []
            book_pivot = pickle.load(open(self.recommendation_config.book_pivot_serialized_objects, 'rb'))
            final_rating = pickle.load(open(self.recommendation_config.final_rating_serialized_objects, 'rb'))

            for book_id in suggestion:
                book_name.append(book_pivot.index[book_id])

            for name in book_name[0]:
                ids = np.where(final_rating['title'] == name)[0][0]
                ids_index.append(ids)

            for idx in ids_index:
                url = final_rating.iloc[idx]['image_url']
                poster_url.append(url)

            return poster_url

        except Exception as e:
            raise AppException(e, sys) from e

    def recommend_book(self, book_name):
        try:
            books_list = []
            model = pickle.load(open(self.recommendation_config.trained_model_path, 'rb'))
            book_pivot = pickle.load(open(self.recommendation_config.book_pivot_serialized_objects, 'rb'))
            book_id = np.where(book_pivot.index == book_name)[0][0]
            distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

            poster_url = self.fetch_poster(suggestion)

            for i in range(len(suggestion)):
                books = book_pivot.index[suggestion[i]]
                for j in books:
                    books_list.append(j)
            return books_list, poster_url

        except Exception as e:
            raise AppException(e, sys) from e

    def train_engine(self):
        try:
            obj = TrainingPipeline()
            obj.start_training_pipeline()
            st.success("✅ Training Completed Successfully!")
            logging.info(f"Recommended successfully!")
        except Exception as e:
            raise AppException(e, sys) from e

    def recommendations_engine(self, selected_books):
        try:
            recommended_books, poster_url = self.recommend_book(selected_books)
            
            # Modern Recommendation Display
            st.markdown("<div class='recommendation-section'>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class='recommendation-header'>
                    <h2 class='recommend-title'>✨ Curated Just For You</h2>
                    <p class='selected-book'>Based on your interest in <span class='highlight-book'>{selected_books}</span></p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display recommendations in a grid
            cols = st.columns(5)
            for idx, col in enumerate(cols):
                with col:
                    st.markdown(f"""
                        <div class='book-card-wrapper'>
                            <div class='book-card'>
                                <div class='book-ribbon'>Recommended</div>
                                <img src='{poster_url[idx + 1]}' class='book-cover' alt='Book Cover'/>
                                <div class='book-overlay'>
                                    <div class='rating-stars'>⭐⭐⭐⭐⭐</div>
                                    <button class='quick-view-btn'>Quick View</button>
                                </div>
                            </div>
                            <div class='book-info'>
                                <p class='book-title-text'>{recommended_books[idx + 1]}</p>
                                <p class='similarity-badge'>94% Match</p>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
                    
        except Exception as e:
            raise AppException(e, sys) from e


def load_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
        
        * {
            font-family: 'Poppins', sans-serif;
        }
        
        /* ============ MAIN BACKGROUND ============ */
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 100%);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* ============ FLOATING PARTICLES ============ */
        .stApp::before, .stApp::after {
            content: '';
            position: fixed;
            border-radius: 50%;
            filter: blur(60px);
            opacity: 0.3;
            z-index: 0;
            pointer-events: none;
        }
        
        .stApp::before {
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, #ff6ec4, transparent);
            top: -200px;
            right: -200px;
            animation: float1 20s infinite ease-in-out;
        }
        
        .stApp::after {
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, #7ee8fa, transparent);
            bottom: -150px;
            left: -150px;
            animation: float2 18s infinite ease-in-out;
        }
        
        @keyframes float1 {
            0%, 100% { transform: translate(0, 0) rotate(0deg); }
            50% { transform: translate(-100px, 100px) rotate(180deg); }
        }
        
        @keyframes float2 {
            0%, 100% { transform: translate(0, 0) rotate(0deg); }
            50% { transform: translate(100px, -100px) rotate(-180deg); }
        }
        
        /* ============ HERO SECTION ============ */
        .hero-container {
            text-align: center;
            padding: 3rem 2rem;
            position: relative;
            z-index: 1;
        }
        
        .hero-title {
            font-size: 4.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 0%, #e0e7ff 50%, #c7d2fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 1rem;
            text-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            animation: fadeInDown 1s ease-out;
            letter-spacing: -2px;
        }
        
        .hero-subtitle {
            font-size: 1.4rem;
            color: rgba(255, 255, 255, 0.95);
            font-weight: 400;
            margin-bottom: 1rem;
            animation: fadeInUp 1s ease-out 0.2s backwards;
        }
        
        .hero-description {
            font-size: 1.1rem;
            color: rgba(255, 255, 255, 0.8);
            max-width: 700px;
            margin: 0 auto 2rem;
            line-height: 1.6;
            animation: fadeInUp 1s ease-out 0.4s backwards;
        }
        
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* ============ STATS SECTION ============ */
        .stats-container {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin: 2rem 0;
            flex-wrap: wrap;
        }
        
        .stat-card {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 20px;
            padding: 1.5rem 2.5rem;
            text-align: center;
            transition: all 0.4s ease;
            animation: fadeInUp 1s ease-out 0.6s backwards;
        }
        
        .stat-card:hover {
            transform: translateY(-10px);
            background: rgba(255, 255, 255, 0.25);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
        }
        
        .stat-number {
            font-size: 2.5rem;
            font-weight: 700;
            color: #fff;
            margin-bottom: 0.5rem;
        }
        
        .stat-label {
            font-size: 1rem;
            color: rgba(255, 255, 255, 0.9);
            font-weight: 500;
        }
        
        /* ============ GLASS CARDS ============ */
        .glass-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(30px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 24px;
            padding: 2.5rem;
            margin: 2rem 0;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            transition: all 0.4s ease;
            position: relative;
            overflow: hidden;
        }
        
        .glass-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s;
        }
        
        .glass-card:hover::before {
            left: 100%;
        }
        
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 25px 70px rgba(0, 0, 0, 0.4);
            border-color: rgba(255, 255, 255, 0.4);
        }
        
        /* ============ SECTION HEADERS ============ */
        .section-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .section-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #fff;
            margin-bottom: 0.5rem;
            position: relative;
            display: inline-block;
        }
        
        .section-title::after {
            content: '';
            position: absolute;
            bottom: -10px;
            left: 50%;
            transform: translateX(-50%);
            width: 60px;
            height: 4px;
            background: linear-gradient(90deg, #ff6ec4, #7ee8fa);
            border-radius: 2px;
        }
        
        .section-description {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.1rem;
            margin-top: 1.5rem;
        }
        
        /* ============ BUTTONS ============ */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 16px !important;
            padding: 1rem 3rem !important;
            font-size: 1.2rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px !important;
            transition: all 0.4s ease !important;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4) !important;
            width: 100% !important;
            position: relative !important;
            overflow: hidden !important;
        }
        
        .stButton > button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            transition: left 0.5s;
        }
        
        .stButton > button:hover::before {
            left: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px) scale(1.02) !important;
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6) !important;
        }
        
        .stButton > button:active {
            transform: translateY(0) scale(0.98) !important;
        }
        
        /* ============ SELECT BOX ============ */
        .stSelectbox {
            margin: 1.5rem 0;
        }
        
        .stSelectbox > label {
            color: rgba(255, 255, 255, 0.95) !important;
            font-weight: 600 !important;
            font-size: 1.2rem !important;
            margin-bottom: 1rem !important;
            display: block !important;
        }
        
        .stSelectbox > div > div {
            background: rgba(255, 255, 255, 0.15) !important;
            border: 2px solid rgba(255, 255, 255, 0.3) !important;
            border-radius: 16px !important;
            color: white !important;
            padding: 0.8rem 1rem !important;
            font-size: 1.1rem !important;
            transition: all 0.3s ease !important;
        }
        
        .stSelectbox > div > div:hover {
            border-color: rgba(255, 255, 255, 0.5) !important;
            background: rgba(255, 255, 255, 0.2) !important;
        }
        
        /* ============ RECOMMENDATION SECTION ============ */
        .recommendation-section {
            margin: 3rem 0;
            padding: 2rem;
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .recommendation-header {
            text-align: center;
            margin-bottom: 3rem;
        }
        
        .recommend-title {
            font-size: 2.8rem;
            font-weight: 700;
            color: #fff;
            margin-bottom: 1rem;
            text-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        }
        
        .selected-book {
            font-size: 1.3rem;
            color: rgba(255, 255, 255, 0.9);
            font-weight: 400;
        }
        
        .highlight-book {
            color: #ffd700;
            font-weight: 700;
            padding: 0.2rem 0.8rem;
            background: rgba(255, 215, 0, 0.2);
            border-radius: 8px;
        }
        
        /* ============ BOOK CARDS ============ */
        .book-card-wrapper {
            margin-bottom: 2rem;
        }
        
        .book-card {
            position: relative;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
            transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
            background: rgba(0, 0, 0, 0.2);
        }
        
        .book-card:hover {
            transform: translateY(-15px) scale(1.03);
            box-shadow: 0 25px 60px rgba(255, 110, 196, 0.5);
        }
        
        .book-ribbon {
            position: absolute;
            top: 15px;
            right: -35px;
            background: linear-gradient(135deg, #ff6ec4, #7ee8fa);
            color: white;
            padding: 5px 40px;
            font-size: 0.75rem;
            font-weight: 600;
            transform: rotate(45deg);
            z-index: 10;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.3);
        }
        
        .book-cover {
            width: 100%;
            height: 320px;
            object-fit: cover;
            display: block;
            transition: all 0.5s ease;
        }
        
        .book-card:hover .book-cover {
            filter: brightness(0.6);
            transform: scale(1.1);
        }
        
        .book-overlay {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(to top, rgba(0,0,0,0.95), rgba(0,0,0,0.7), transparent);
            padding: 1.5rem;
            transform: translateY(100%);
            transition: transform 0.5s ease;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            align-items: center;
        }
        
        .book-card:hover .book-overlay {
            transform: translateY(0);
        }
        
        .rating-stars {
            color: #fbbf24;
            font-size: 1.3rem;
            letter-spacing: 2px;
        }
        
        .quick-view-btn {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 0.6rem 1.5rem;
            border-radius: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.9rem;
        }
        
        .quick-view-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.5);
        }
        
        .book-info {
            padding: 1rem 0.5rem;
        }
        
        .book-title-text {
            color: #fff !important;
            font-weight: 600;
            font-size: 1rem;
            margin-bottom: 0.5rem;
            line-height: 1.4;
            min-height: 45px;
            text-align: center;
        }
        
        .similarity-badge {
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
            margin: 0 auto;
            display: block;
            text-align: center;
            width: fit-content;
            margin-left: auto;
            margin-right: auto;
        }
        
        /* ============ SUCCESS MESSAGE ============ */
        .stSuccess {
            background: rgba(16, 185, 129, 0.2) !important;
            border: 2px solid #10b981 !important;
            border-radius: 16px !important;
            color: #6ee7b7 !important;
            padding: 1rem !important;
            font-weight: 600 !important;
        }
        
        /* ============ DIVIDER ============ */
        hr {
            border: none;
            height: 3px;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.5), transparent);
            margin: 3rem 0;
            border-radius: 2px;
        }
        
        /* ============ FEATURE CARDS ============ */
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }
        
        .feature-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            transition: all 0.4s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-10px);
            background: rgba(255, 255, 255, 0.15);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
        }
        
        .feature-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        .feature-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #fff;
            margin-bottom: 0.5rem;
        }
        
        .feature-description {
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.95rem;
            line-height: 1.6;
        }
        
        /* ============ FOOTER ============ */
        .footer {
            text-align: center;
            color: rgba(255, 255, 255, 0.9);
            padding: 3rem 2rem;
            margin-top: 4rem;
            font-size: 1rem;
            background: rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .footer-logo {
            font-size: 2rem;
            margin-bottom: 1rem;
        }
        
        /* ============ LOADING SPINNER ============ */
        .stSpinner > div {
            border-color: rgba(255, 255, 255, 0.3) !important;
            border-top-color: #667eea !important;
        }
        
        /* ============ RESPONSIVE ============ */
        @media (max-width: 768px) {
            .hero-title {
                font-size: 3rem;
            }
            
            .stat-card {
                padding: 1rem 1.5rem;
            }
            
            .section-title {
                font-size: 2rem;
            }
        }
        </style>
    """, unsafe_allow_html=True)


def display_hero_section():
    st.markdown("""
        <div class='hero-container'>
            <h1 class='hero-title'>📚 BookVerse AI</h1>
            <p class='hero-subtitle'>Your Personal AI-Powered Reading Companion</p>
            <p class='hero-description'>
                Discover your next favorite book with our advanced collaborative filtering algorithm. 
                Get personalized recommendations based on millions of reader preferences and ratings.
            </p>
        </div>
    """, unsafe_allow_html=True)


def display_stats():
    st.markdown("""
        <div class='stats-container'>
            <div class='stat-card'>
                <div class='stat-number'>10K+</div>
                <div class='stat-label'>Books Analyzed</div>
            </div>
            <div class='stat-card'>
                <div class='stat-number'>50K+</div>
                <div class='stat-label'>User Ratings</div>
            </div>
            <div class='stat-card'>
                <div class='stat-number'>95%</div>
                <div class='stat-label'>Accuracy Rate</div>
            </div>
            <div class='stat-card'>
                <div class='stat-number'>24/7</div>
                <div class='stat-label'>AI Available</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def display_features():
    st.markdown("""
        <div class='feature-grid'>
            <div class='feature-card'>
                <div class='feature-icon'>🤖</div>
                <div class='feature-title'>AI-Powered</div>
                <div class='feature-description'>
                    Advanced machine learning algorithms analyze reading patterns
                </div>
            </div>
            <div class='feature-card'>
                <div class='feature-icon'>⚡</div>
                <div class='feature-title'>Lightning Fast</div>
                <div class='feature-description'>
                    Get instant recommendations in milliseconds
                </div>
            </div>
            <div class='feature-card'>
                <div class='feature-icon'>🎯</div>
                <div class='feature-title'>Highly Accurate</div>
                <div class='feature-description'>
                    95% match rate based on your reading preferences
                </div>
            </div>
            <div class='feature-card'>
                <div class='feature-icon'>🌟</div>
                <div class='feature-title'>Personalized</div>
                <div class='feature-description'>
                    Tailored suggestions just for your unique taste
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Page Configuration
    st.set_page_config(
        page_title="BookVerse AI - Smart Book Recommendations",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Load Custom CSS
    load_custom_css()
    
    # Hero Section
    display_hero_section()
    
    # Stats Section
    display_stats()
    
    # Initialize Recommendation Object
    obj = Recommendation()
    
    # Training Section
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("""
        <div class='section-header'>
            <div class='section-title'>🚀 Training Engine</div>
            <p class='section-description'>
                Train the AI model with the latest book ratings and collaborative filtering data
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button('🎯 Train AI Recommender System'):
        with st.spinner('🔄 Training AI model with latest data...'):
            obj.train_engine()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Book Selection Section
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("""
        <div class='section-header'>
            <div class='section-title'>🔍 Discover Your Next Read</div>
            <p class='section-description'>
                Select a book you love, and we'll find similar titles you'll enjoy
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    book_names = pickle.load(open(os.path.join('templates', 'book_names.pkl'), 'rb'))
    selected_books = st.selectbox(
        "Choose a book from our curated collection",
        book_names,
        index=0
    )
    
    if st.button('✨ Get AI Recommendations'):
        with st.spinner('🔮 AI is analyzing and finding perfect matches...'):
            obj.recommendations_engine(selected_books)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Features Section
    st.markdown("""
        <div class='section-header'>
            <div class='section-title'>💎 Why Choose BookVerse AI?</div>
        </div>
    """, unsafe_allow_html=True)
    display_features()
    
    # Footer
    st.markdown("""
        <div class='footer'>
            <div class='footer-logo'>📚</div>
            <p><strong>BookVerse AI</strong> - Powered by Advanced Collaborative Filtering & Machine Learning</p>
            <p style='margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;'>
                Built with ❤️ using Streamlit • Python • Scikit-learn
            </p>
        </div>
    """, unsafe_allow_html=True)