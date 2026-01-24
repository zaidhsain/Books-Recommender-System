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
            
            st.markdown("---")
            st.markdown(f"<h2 class='recommend-title'>📚 Recommended For You</h2>", unsafe_allow_html=True)
            st.markdown(f"<p class='selected-book'>Based on: <span class='highlight'>{selected_books}</span></p>", unsafe_allow_html=True)
            
            cols = st.columns(5)
            for idx, col in enumerate(cols):
                with col:
                    st.markdown(f"""
                        <div class='book-card'>
                            <img src='{poster_url[idx + 1]}' class='book-image'/>
                            <div class='book-overlay'>
                                <div class='stars'>⭐⭐⭐⭐⭐</div>
                            </div>
                        </div>
                        <p class='book-title'>{recommended_books[idx + 1]}</p>
                    """, unsafe_allow_html=True)
                    
        except Exception as e:
            raise AppException(e, sys) from e


# Custom CSS Styling
def load_custom_css():
    st.markdown("""
        <style>
        /* Main Background with Gradient */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0d9488 100%);
            background-attachment: fixed;
        }
        
        /* Animated Background Orbs */
        .stApp::before {
            content: '';
            position: fixed;
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, rgba(6, 182, 212, 0.15) 0%, transparent 70%);
            top: -100px;
            left: -100px;
            animation: float 20s infinite ease-in-out;
            z-index: 0;
            pointer-events: none;
        }
        
        .stApp::after {
            content: '';
            position: fixed;
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(16, 185, 129, 0.15) 0%, transparent 70%);
            bottom: -100px;
            right: -100px;
            animation: float 15s infinite ease-in-out reverse;
            z-index: 0;
            pointer-events: none;
        }
        
        @keyframes float {
            0%, 100% { transform: translate(0, 0) scale(1); }
            33% { transform: translate(50px, -50px) scale(1.1); }
            66% { transform: translate(-30px, 30px) scale(0.9); }
        }
        
        /* Header Styling */
        h1 {
            background: linear-gradient(90deg, #06b6d4, #3b82f6, #10b981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3.5rem !important;
            font-weight: 800 !important;
            text-align: center;
            margin-bottom: 0.5rem !important;
            text-shadow: 0 0 30px rgba(6, 182, 212, 0.3);
        }
        
        /* Subtitle */
        .subtitle {
            text-align: center;
            color: #a5f3fc;
            font-size: 1.2rem;
            margin-bottom: 2rem;
            font-weight: 300;
        }
        
        /* Cards with Glassmorphism */
        .css-1r6slb0, .css-12oz5g7 {
            background: rgba(255, 255, 255, 0.08) !important;
            backdrop-filter: blur(20px) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 20px !important;
            padding: 2rem !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
            margin-bottom: 2rem !important;
        }
        
        /* Section Headers */
        h2 {
            color: #06b6d4 !important;
            font-weight: 700 !important;
            font-size: 1.8rem !important;
            margin-bottom: 1rem !important;
        }
        
        h3 {
            color: #67e8f9 !important;
            font-weight: 600 !important;
            margin-bottom: 1rem !important;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #06b6d4, #3b82f6) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.75rem 2rem !important;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(6, 182, 212, 0.4) !important;
            width: 100% !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) scale(1.02) !important;
            box-shadow: 0 6px 25px rgba(6, 182, 212, 0.6) !important;
            background: linear-gradient(135deg, #0891b2, #2563eb) !important;
        }
        
        .stButton > button:active {
            transform: translateY(0) scale(0.98) !important;
        }
        
        /* Select Box */
        .stSelectbox > div > div {
            background: rgba(255, 255, 255, 0.1) !important;
            border: 1px solid rgba(6, 182, 212, 0.3) !important;
            border-radius: 12px !important;
            color: white !important;
            padding: 0.5rem !important;
        }
        
        .stSelectbox label {
            color: #a5f3fc !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
        }
        
        /* Text Color */
        p, label, .stMarkdown {
            color: #e0f2fe !important;
        }
        
        /* Book Recommendations */
        .recommend-title {
            background: linear-gradient(90deg, #06b6d4, #10b981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.2rem;
            font-weight: 700;
            text-align: center;
            margin: 2rem 0 1rem 0;
        }
        
        .selected-book {
            text-align: center;
            color: #bae6fd;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        
        .highlight {
            color: #06b6d4;
            font-weight: 700;
        }
        
        /* Book Card */
        .book-card {
            position: relative;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
            transition: all 0.4s ease;
            margin-bottom: 1rem;
            cursor: pointer;
        }
        
        .book-card:hover {
            transform: translateY(-10px) scale(1.05);
            box-shadow: 0 15px 40px rgba(6, 182, 212, 0.5);
        }
        
        .book-image {
            width: 100%;
            height: 300px;
            object-fit: cover;
            display: block;
            transition: all 0.4s ease;
        }
        
        .book-card:hover .book-image {
            filter: brightness(0.7);
        }
        
        .book-overlay {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(to top, rgba(0,0,0,0.9), transparent);
            padding: 1rem;
            opacity: 0;
            transition: opacity 0.4s ease;
        }
        
        .book-card:hover .book-overlay {
            opacity: 1;
        }
        
        .stars {
            color: #fbbf24;
            font-size: 1.2rem;
            text-align: center;
        }
        
        .book-title {
            text-align: center;
            color: #e0f2fe !important;
            font-weight: 600;
            font-size: 0.95rem;
            margin-top: 0.5rem;
            line-height: 1.4;
            min-height: 40px;
            transition: color 0.3s ease;
        }
        
        .book-title:hover {
            color: #06b6d4 !important;
        }
        
        /* Success Message */
        .stSuccess {
            background: rgba(16, 185, 129, 0.2) !important;
            border: 1px solid #10b981 !important;
            border-radius: 12px !important;
            color: #6ee7b7 !important;
        }
        
        /* Divider */
        hr {
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, #06b6d4, transparent);
            margin: 2rem 0;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            color: #67e8f9;
            padding: 2rem;
            margin-top: 3rem;
            font-size: 0.95rem;
        }
        
        /* Icon Styling */
        .icon {
            display: inline-block;
            margin-right: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Page Configuration
    st.set_page_config(
        page_title="BookVerse - AI Recommender",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Load Custom CSS
    load_custom_css()
    
    # Header
    st.markdown("<h1>📚 BookVerse</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Discover your next favorite read with AI-powered collaborative filtering</p>", unsafe_allow_html=True)
    
    obj = Recommendation()
    
    # Training Section
    st.markdown("<h2>🚀 Training Engine</h2>", unsafe_allow_html=True)
    st.markdown("<p>Train the recommendation model with the latest book ratings and user preferences</p>", unsafe_allow_html=True)
    
    if st.button('🎯 Train Recommender System'):
        with st.spinner('🔄 Training in progress...'):
            obj.train_engine()
    
    st.markdown("---")
    
    # Book Selection Section
    st.markdown("<h2>🔍 Find Your Book</h2>", unsafe_allow_html=True)
    
    book_names = pickle.load(open(os.path.join('templates', 'book_names.pkl'), 'rb'))
    selected_books = st.selectbox(
        "Type or select a book from the dropdown",
        book_names,
        index=0
    )
    
    # Recommendation Button
    if st.button('✨ Show Recommendations'):
        with st.spinner('🔮 Finding perfect matches...'):
            obj.recommendations_engine(selected_books)
    
    # Footer
    st.markdown("<div class='footer'>Powered by Collaborative Filtering Algorithm ✨</div>", unsafe_allow_html=True)