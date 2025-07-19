import sqlite3
import json
import pandas as pd
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PromptDatabase:
    def __init__(self, db_path: str = "promptcraft.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create prompts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_goal TEXT NOT NULL,
                    generated_prompt TEXT NOT NULL,
                    rating INTEGER,
                    feedback TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    category TEXT,
                    tags TEXT
                )
            ''')
            
            # Create usage analytics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_goal TEXT NOT NULL,
                    prompt_variations_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def save_prompt(self, user_goal: str, generated_prompt: str, rating: int = None, 
                   feedback: str = None, category: str = None, tags: List[str] = None):
        """Save a prompt to the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            tags_str = json.dumps(tags) if tags else None
            
            cursor.execute('''
                INSERT INTO prompts (user_goal, generated_prompt, rating, feedback, category, tags)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_goal, generated_prompt, rating, feedback, category, tags_str))
            
            conn.commit()
    
    def get_similar_prompts(self, user_goal: str, top_k: int = 3) -> List[Dict]:
        """Retrieve similar high-rated prompts using enhanced RAG with multiple similarity methods"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get all prompts with ratings >= 4 (high quality prompts for RAG)
            cursor.execute('''
                SELECT user_goal, generated_prompt, rating, feedback, category, tags
                FROM prompts 
                WHERE rating >= 4
                ORDER BY rating DESC, created_at DESC
            ''')
            
            results = cursor.fetchall()
        
        if not results:
            return []
        
        prompts_data = [
            {
                'goal': row[0],
                'prompt': row[1],
                'rating': row[2],
                'feedback': row[3],
                'category': row[4],
                'tags': json.loads(row[5]) if row[5] else []
            }
            for row in results
        ]
        
        try:
            goals = [row[0] for row in results]
            if not goals:
                return []

            # Method 1: TF-IDF similarity on goals
            vectorizer = TfidfVectorizer(
                stop_words='english', 
                max_features=1000,
                ngram_range=(1, 2)
            )
            all_goals = goals + [user_goal]
            tfidf_matrix = vectorizer.fit_transform(all_goals)
            
            goal_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
            
            # Method 2: Category matching bonus
            user_words = set(user_goal.lower().split())
            category_bonuses = []
            
            for prompt_data in prompts_data:
                bonus = 0
                if prompt_data['category']:
                    category_words = set(prompt_data['category'].lower().split())
                    if category_words.intersection(user_words):
                        bonus += 0.2
                
                if prompt_data['tags']:
                    tag_words = set(' '.join(prompt_data['tags']).lower().split())
                    common_words = user_words.intersection(tag_words)
                    bonus += len(common_words) * 0.1
                
                category_bonuses.append(bonus)
            
            # Combine similarities with bonuses and rating weights
            final_scores = []
            for i, (sim, bonus, prompt_data) in enumerate(zip(goal_similarities, category_bonuses, prompts_data)):
                rating_weight = prompt_data['rating'] / 5.0
                final_score = (sim * 0.7) + (bonus * 0.2) + (rating_weight * 0.1)
                final_scores.append((final_score, i, prompt_data))
            
            final_scores.sort(reverse=True, key=lambda x: x[0])
            
            relevant_prompts = [
                prompt_data for score, idx, prompt_data in final_scores 
                if score > 0.1
            ]
            
            return relevant_prompts[:top_k]
            
        except Exception as e:
            logging.warning(f"RAG similarity calculation failed, using fallback: {str(e)}")
            return self._fallback_similarity_search(user_goal, prompts_data, top_k)
    
    def _fallback_similarity_search(self, user_goal: str, prompts_data: List[Dict], top_k: int) -> List[Dict]:
        """Fallback similarity search using simple keyword matching"""
        user_words = set(user_goal.lower().split())
        
        scored_prompts = []
        for prompt_data in prompts_data:
            goal_words = set(prompt_data['goal'].lower().split())
            common_words = len(user_words.intersection(goal_words))
            # Avoid division by zero
            denominator = max(len(user_words), len(goal_words), 1)
            score = common_words / denominator
            scored_prompts.append((score, prompt_data))
        
        scored_prompts.sort(reverse=True, key=lambda x: x[0])
        return [prompt_data for score, prompt_data in scored_prompts[:top_k] if score > 0]
    
    def log_usage(self, user_goal: str, prompt_count: int):
        """Log usage analytics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO analytics (user_goal, prompt_variations_count)
                VALUES (?, ?)
            ''', (user_goal, prompt_count))
            
            conn.commit()
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM prompts')
            total_prompts = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM prompts WHERE rating IS NOT NULL')
            rated_prompts = cursor.fetchone()[0]
            
            cursor.execute('SELECT AVG(rating) FROM prompts WHERE rating IS NOT NULL')
            avg_rating = cursor.fetchone()[0] or 0
            
            cursor.execute('''
                SELECT category, COUNT(*) as count 
                FROM prompts 
                WHERE category IS NOT NULL 
                GROUP BY category 
                ORDER BY count DESC 
                LIMIT 5
            ''')
            top_categories = cursor.fetchall()
        
        return {
            'total_prompts': total_prompts,
            'rated_prompts': rated_prompts,
            'avg_rating': round(avg_rating, 2),
            'top_categories': top_categories
        }