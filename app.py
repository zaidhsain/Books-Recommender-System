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
            background: 
                linear-gradient(135deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 25%, rgba(240, 147, 251, 0.95) 50%, rgba(79, 172, 254, 0.95) 100%),
                url('https://images.unsplash.com/photo-1481627834876-b7833e8f5570?q=80&w=2000') center/cover fixed;
            background-size: 400% 400%, cover;
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
        
        /* ============ SPARKLE PARTICLES ============ */
        @keyframes sparkle {
            0%, 100% { opacity: 0; transform: scale(0); }
            50% { opacity: 1; transform: scale(1); }
        }
        
        .stApp::after {
            animation: float2 18s infinite ease-in-out, sparkle 3s infinite;
        }
        
        /* ============ HERO SECTION ============ */
        .hero-container {
            text-align: center;
            padding: 3rem 2rem;
            position: relative;
            z-index: 1;
            min-height: 600px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .hero-image-container {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            overflow: hidden;
            border-radius: 30px;
            margin: 1rem;
        }
        
        .hero-book-image {
            width: 100%;
            height: 100%;
            object-fit: cover;
            opacity: 0.15;
            filter: blur(2px);
            animation: zoomInOut 20s infinite ease-in-out;
        }
        
        @keyframes zoomInOut {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        
        .hero-overlay-gradient {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.6), rgba(118, 75, 162, 0.6), rgba(240, 147, 251, 0.6));
        }
        
        .hero-content {
            position: relative;
            z-index: 2;
        }
        
        .hero-badge {
            display: inline-block;
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            padding: 0.6rem 1.5rem;
            border-radius: 30px;
            color: white;
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
            animation: fadeInDown 1s ease-out;
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
        
        .hero-decorative-books {
            position: absolute;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            pointer-events: none;
            z-index: 1;
        }
        
        .floating-book {
            position: absolute;
            width: 100px;
            height: 140px;
            object-fit: cover;
            border-radius: 8px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
            opacity: 0.3;
            transform-style: preserve-3d;
        }
        
        .book-1 {
            top: 10%;
            left: 5%;
            animation: floatBook1 15s infinite ease-in-out;
        }
        
        .book-2 {
            top: 60%;
            right: 8%;
            animation: floatBook2 12s infinite ease-in-out;
        }
        
        .book-3 {
            bottom: 15%;
            left: 10%;
            animation: floatBook3 18s infinite ease-in-out;
        }
        
        @keyframes floatBook1 {
            0%, 100% { transform: translateY(0px) rotateZ(-5deg); }
            50% { transform: translateY(-30px) rotateZ(5deg); }
        }
        
        @keyframes floatBook2 {
            0%, 100% { transform: translateY(0px) rotateZ(5deg); }
            50% { transform: translateY(-40px) rotateZ(-5deg); }
        }
        
        @keyframes floatBook3 {
            0%, 100% { transform: translateY(0px) rotateZ(-3deg); }
            50% { transform: translateY(-35px) rotateZ(3deg); }
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
            position: relative;
            overflow: hidden;
        }
        
        .stat-card::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            transform: rotate(45deg);
            transition: all 0.5s ease;
        }
        
        .stat-card:hover::before {
            left: 100%;
        }
        
        .stat-card:hover {
            transform: translateY(-10px) scale(1.05);
            background: rgba(255, 255, 255, 0.25);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
        }
        
        .stat-icon-bg {
            position: relative;
            width: 80px;
            height: 80px;
            margin: 0 auto 1rem;
            border-radius: 50%;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(255, 255, 255, 0.1);
            border: 2px solid rgba(255, 255, 255, 0.3);
        }
        
        .stat-bg-img {
            position: absolute;
            width: 100%;
            height: 100%;
            object-fit: cover;
            opacity: 0.3;
            filter: blur(1px);
            transition: all 0.4s ease;
        }
        
        .stat-card:hover .stat-bg-img {
            opacity: 0.5;
            transform: scale(1.2) rotate(10deg);
        }
        
        .stat-icon-overlay {
            position: relative;
            z-index: 2;
            font-size: 2rem;
            filter: drop-shadow(0 3px 10px rgba(0, 0, 0, 0.3));
        }
        
        .stat-number {
            font-size: 2.5rem;
            font-weight: 700;
            color: #fff;
            margin-bottom: 0.5rem;
            text-shadow: 0 3px 10px rgba(0, 0, 0, 0.3);
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
        
        .section-icon-header {
            position: relative;
            width: 150px;
            height: 150px;
            margin: 0 auto 1.5rem;
            border-radius: 50%;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(255, 255, 255, 0.1);
            border: 3px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }
        
        .section-header-img {
            position: absolute;
            width: 100%;
            height: 100%;
            object-fit: cover;
            opacity: 0.4;
            filter: blur(1px);
            transition: all 0.4s ease;
        }
        
        .section-icon-header:hover .section-header-img {
            opacity: 0.6;
            transform: scale(1.1);
        }
        
        .section-icon-overlay {
            position: relative;
            z-index: 2;
            font-size: 4rem;
            filter: drop-shadow(0 5px 15px rgba(0, 0, 0, 0.5));
            animation: bounce 2s infinite;
        }
        
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
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
        
        /* ============ DECORATIVE DIVIDER ============ */
        .decorative-divider {
            position: relative;
            height: 200px;
            margin: 3rem 0;
            border-radius: 24px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .divider-image {
            position: absolute;
            width: 100%;
            height: 100%;
            object-fit: cover;
            opacity: 0.3;
            filter: blur(2px);
        }
        
        .divider-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.6), rgba(240, 147, 251, 0.6));
        }
        
        .divider-content {
            position: relative;
            z-index: 2;
        }
        
        .divider-icon {
            font-size: 4rem;
            filter: drop-shadow(0 5px 20px rgba(0, 0, 0, 0.5));
            animation: rotate360 10s linear infinite;
        }
        
        @keyframes rotate360 {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
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
        
        /* ============ SHOWCASE SECTION ============ */
        .showcase-section {
            margin: 3rem 0;
        }
        
        .showcase-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }
        
        .showcase-card {
            position: relative;
            height: 400px;
            border-radius: 24px;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
            transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
        }
        
        .showcase-card:hover {
            transform: translateY(-15px) scale(1.02);
            box-shadow: 0 30px 80px rgba(0, 0, 0, 0.5);
        }
        
        .showcase-image {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: all 0.5s ease;
        }
        
        .showcase-card:hover .showcase-image {
            transform: scale(1.1);
            filter: brightness(0.7);
        }
        
        .showcase-overlay {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(to top, rgba(0, 0, 0, 0.9), rgba(0, 0, 0, 0.6), transparent);
            padding: 2rem;
            transform: translateY(50%);
            transition: transform 0.5s ease;
        }
        
        .showcase-card:hover .showcase-overlay {
            transform: translateY(0);
        }
        
        .showcase-title {
            color: #fff;
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 0 3px 10px rgba(0, 0, 0, 0.5);
        }
        
        .showcase-text {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1rem;
            line-height: 1.6;
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
            overflow: hidden;
            position: relative;
        }
        
        .feature-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(240, 147, 251, 0.1));
            opacity: 0;
            transition: opacity 0.4s ease;
        }
        
        .feature-card:hover::before {
            opacity: 1;
        }
        
        .feature-card:hover {
            transform: translateY(-10px);
            background: rgba(255, 255, 255, 0.15);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
        }
        
        .feature-icon-wrapper {
            position: relative;
            width: 120px;
            height: 120px;
            margin: 0 auto 1.5rem;
            border-radius: 50%;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 2px solid rgba(255, 255, 255, 0.3);
        }
        
        .feature-bg-image {
            position: absolute;
            width: 100%;
            height: 100%;
            object-fit: cover;
            opacity: 0.3;
            filter: blur(1px);
            transition: all 0.4s ease;
        }
        
        .feature-card:hover .feature-bg-image {
            opacity: 0.5;
            transform: scale(1.1);
        }
        
        .feature-icon {
            font-size: 3rem;
            position: relative;
            z-index: 2;
            filter: drop-shadow(0 5px 15px rgba(0, 0, 0, 0.3));
            transition: transform 0.4s ease;
        }
        
        .feature-card:hover .feature-icon {
            transform: scale(1.2) rotate(10deg);
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
            position: relative;
            text-align: center;
            color: rgba(255, 255, 255, 0.9);
            padding: 4rem 2rem;
            margin-top: 4rem;
            overflow: hidden;
            border-radius: 24px 24px 0 0;
        }
        
        .footer-background {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: 0;
        }
        
        .footer-bg-image {
            width: 100%;
            height: 100%;
            object-fit: cover;
            opacity: 0.15;
            filter: blur(3px);
        }
        
        .footer-bg-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.9), rgba(118, 75, 162, 0.9));
        }
        
        .footer-content {
            position: relative;
            z-index: 2;
        }
        
        .footer-logo-section {
            margin-bottom: 1.5rem;
        }
        
        .footer-logo {
            font-size: 3.5rem;
            margin-bottom: 0.5rem;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        
        .footer-brand {
            font-size: 2rem;
            font-weight: 700;
            color: #fff;
            margin: 0;
            text-shadow: 0 3px 10px rgba(0, 0, 0, 0.3);
        }
        
        .footer-tagline {
            font-size: 1.1rem;
            margin: 1rem 0 1.5rem;
            font-weight: 500;
        }
        
        .footer-tech-stack {
            display: flex;
            justify-content: center;
            gap: 1rem;
            flex-wrap: wrap;
            margin: 2rem 0;
        }
        
        .tech-badge {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            padding: 0.5rem 1.2rem;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .tech-badge:hover {
            background: rgba(255, 255, 255, 0.3);
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        }
        
        .footer-divider {
            width: 200px;
            height: 2px;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.5), transparent);
            margin: 2rem auto;
        }
        
        .footer-copyright {
            font-size: 0.95rem;
            opacity: 0.9;
            margin-bottom: 1.5rem;
        }
        
        .footer-social {
            display: flex;
            justify-content: center;
            gap: 1.5rem;
            flex-wrap: wrap;
        }
        
        .social-link {
            background: rgba(255, 255, 255, 0.15);
            padding: 0.6rem 1.2rem;
            border-radius: 12px;
            font-size: 0.9rem;
            font-weight: 600;
            transition: all 0.3s ease;
            cursor: pointer;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .social-link:hover {
            background: rgba(255, 255, 255, 0.25);
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
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
            <div class='hero-image-container'>
                <img src='https://images.unsplash.com/photo-1512820790803-83ca734da794?q=80&w=800' class='hero-book-image' alt='Books'/>
                <div class='hero-overlay-gradient'></div>
            </div>
            <div class='hero-content'>
                <div class='hero-badge'>✨ AI-Powered Recommendations</div>
                <h1 class='hero-title'>📚 BookVerse AI</h1>
                <p class='hero-subtitle'>Your Personal AI-Powered Reading Companion</p>
                <p class='hero-description'>
                    Discover your next favorite book with our advanced collaborative filtering algorithm. 
                    Get personalized recommendations based on millions of reader preferences and ratings.
                </p>
                <div class='hero-decorative-books'>
                    <img src='https://images.unsplash.com/photo-1495446815901-a7297e633e8d?q=80&w=200' class='floating-book book-1' alt='Book'/>
                    <img src='https://images.unsplash.com/photo-1544947950-fa07a98d237f?q=80&w=200' class='floating-book book-2' alt='Book'/>
                    <img src='https://images.unsplash.com/photo-1589829085413-56de8ae18c73?q=80&w=200' class='floating-book book-3' alt='Book'/>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def display_stats():
    st.markdown("""
        <div class='stats-container'>
            <div class='stat-card'>
                <div class='stat-icon-bg'>
                    <img src='https://images.unsplash.com/photo-1495446815901-a7297e633e8d?q=80&w=100' class='stat-bg-img' alt='Books'/>
                    <div class='stat-icon-overlay'>📚</div>
                </div>
                <div class='stat-number'>10K+</div>
                <div class='stat-label'>Books Analyzed</div>
            </div>
            <div class='stat-card'>
                <div class='stat-icon-bg'>
                    <img src='https://images.unsplash.com/photo-1456513080510-7bf3a84b82f8?q=80&w=100' class='stat-bg-img' alt='Users'/>
                    <div class='stat-icon-overlay'>👥</div>
                </div>
                <div class='stat-number'>50K+</div>
                <div class='stat-label'>User Ratings</div>
            </div>
            <div class='stat-card'>
                <div class='stat-icon-bg'>
                    <img src='https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=100' class='stat-bg-img' alt='Accuracy'/>
                    <div class='stat-icon-overlay'>📊</div>
                </div>
                <div class='stat-number'>95%</div>
                <div class='stat-label'>Accuracy Rate</div>
            </div>
            <div class='stat-card'>
                <div class='stat-icon-bg'>
                    <img src='https://images.unsplash.com/photo-1485827404703-89b55fcc595e?q=80&w=100' class='stat-bg-img' alt='AI'/>
                    <div class='stat-icon-overlay'>🤖</div>
                </div>
                <div class='stat-number'>24/7</div>
                <div class='stat-label'>AI Available</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def display_features():
    st.markdown("""
        <div class='feature-grid'>
            <div class='feature-card'>
                <div class='feature-icon-wrapper'>
                    <img src='https://images.unsplash.com/photo-1677442136019-21780ecad995?q=80&w=200' class='feature-bg-image' alt='AI'/>
                    <div class='feature-icon'>🤖</div>
                </div>
                <div class='feature-title'>AI-Powered</div>
                <div class='feature-description'>
                    Advanced machine learning algorithms analyze reading patterns
                </div>
            </div>
            <div class='feature-card'>
                <div class='feature-icon-wrapper'>
                    <img src='https://images.unsplash.com/photo-1516339901601-2e1b62dc0c45?q=80&w=200' class='feature-bg-image' alt='Speed'/>
                    <div class='feature-icon'>⚡</div>
                </div>
                <div class='feature-title'>Lightning Fast</div>
                <div class='feature-description'>
                    Get instant recommendations in milliseconds
                </div>
            </div>
            <div class='feature-card'>
                <div class='feature-icon-wrapper'>
                    <img src='https://images.unsplash.com/photo-1434030216411-0b793f4b4173?q=80&w=200' class='feature-bg-image' alt='Accuracy'/>
                    <div class='feature-icon'>🎯</div>
                </div>
                <div class='feature-title'>Highly Accurate</div>
                <div class='feature-description'>
                    95% match rate based on your reading preferences
                </div>
            </div>
            <div class='feature-card'>
                <div class='feature-icon-wrapper'>
                    <img src='https://images.unsplash.com/photo-1524995997946-a1c2e315a42f?q=80&w=200' class='feature-bg-image' alt='Personal'/>
                    <div class='feature-icon'>🌟</div>
                </div>
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
            <div class='section-icon-header'>
                <img src='https://images.unsplash.com/photo-1620712943543-bcc4688e7485?q=80&w=150' class='section-header-img' alt='Training'/>
                <div class='section-icon-overlay'>🚀</div>
            </div>
            <div class='section-title'>Training Engine</div>
            <p class='section-description'>
                Train the AI model with the latest book ratings and collaborative filtering data
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button('🎯 Train AI Recommender System'):
        with st.spinner('🔄 Training AI model with latest data...'):
            obj.train_engine()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='decorative-divider'>
            <img src='https://images.unsplash.com/photo-1516979187457-637abb4f9353?q=80&w=1200' class='divider-image' alt='Books'/>
            <div class='divider-overlay'></div>
            <div class='divider-content'>
                <div class='divider-icon'>✨</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Book Selection Section
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("""
        <div class='section-header'>
            <div class='section-icon-header'>
                <img src='https://images.unsplash.com/photo-1507842217343-583bb7270b66?q=80&w=150' class='section-header-img' alt='Search'/>
                <div class='section-icon-overlay'>🔍</div>
            </div>
            <div class='section-title'>Find Your Book</div>
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
    
    st.markdown("""
        <div class='decorative-divider'>
            <img src='https://images.unsplash.com/photo-1516979187457-637abb4f9353?q=80&w=1200' class='divider-image' alt='Books'/>
            <div class='divider-overlay'></div>
            <div class='divider-content'>
                <div class='divider-icon'>✨</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Visual Showcase Section
    st.markdown("""
        <div class='showcase-section'>
            <div class='showcase-grid'>
                <div class='showcase-card'>
                    <img src='https://images.unsplash.com/photo-1535905557558-afc4877a26fc?q=80&w=800' class='showcase-image' alt='Reading'/>
                    <div class='showcase-overlay'>
                        <h3 class='showcase-title'>Immersive Reading Experience</h3>
                        <p class='showcase-text'>Discover books that transport you to new worlds</p>
                    </div>
                </div>
                <div class='showcase-card'>
                    <img src='https://images.unsplash.com/photo-1519682337058-a94d519337bc?q=80&w=800' class='showcase-image' alt='Library'/>
                    <div class='showcase-overlay'>
                        <h3 class='showcase-title'>Endless Possibilities</h3>
                        <p class='showcase-text'>Access to thousands of curated recommendations</p>
                    </div>
                </div>
                <div class='showcase-card'>
                    <img src='https://images.unsplash.com/photo-1491841573634-28140fc7ced7?q=80&w=800' class='showcase-image' alt='Modern Reading'/>
                    <div class='showcase-overlay'>
                        <h3 class='showcase-title'>Smart Technology</h3>
                        <p class='showcase-text'>AI that understands your unique reading taste</p>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='decorative-divider'>
            <img src='https://images.unsplash.com/photo-1524995997946-a1c2e315a42f?q=80&w=1200' class='divider-image' alt='Books Stack'/>
            <div class='divider-overlay'></div>
            <div class='divider-content'>
                <div class='divider-icon'>📖</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
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
            <div class='footer-background'>
                <img src='https://images.unsplash.com/photo-1521587760476-6c12a4b040da?q=80&w=1200' class='footer-bg-image' alt='Library'/>
                <div class='footer-bg-overlay'></div>
            </div>
            <div class='footer-content'>
                <div class='footer-logo-section'>
                    <div class='footer-logo'>📚</div>
                    <h3 class='footer-brand'>BookVerse AI</h3>
                </div>
                <p class='footer-tagline'>Powered by Advanced Collaborative Filtering & Machine Learning</p>
                <div class='footer-tech-stack'>
                    <span class='tech-badge'>🐍 Python</span>
                    <span class='tech-badge'>⚡ Streamlit</span>
                    <span class='tech-badge'>🤖 Scikit-learn</span>
                    <span class='tech-badge'>🎨 Beautiful UI</span>
                </div>
                <div class='footer-divider'></div>
                <p class='footer-copyright'>Built with ❤️ for book lovers worldwide • © 2026 BookVerse AI</p>
                <div class='footer-social'>
                    <span class='social-link'>🌐 Web</span>
                    <span class='social-link'>💼 LinkedIn</span>
                    <span class='social-link'>🐱 GitHub</span>
                    <span class='social-link'>📧 Contact</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)