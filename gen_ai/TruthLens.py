import requests
from google.generativeai import GenerativeModel, configure  # type: ignore
import json
import streamlit as st
import PIL.Image
import io
import sqlite3
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from fastembed import TextEmbedding
from datetime import datetime
import hashlib
from typing import Optional

def configure_genai(api_key: str):
    """
    Configures the Google Generative AI client with the provided API key
    and returns a model instance.
    """
    try:
        configure(api_key=api_key)
        model = GenerativeModel("gemini-2.5-pro")
        return model
    except Exception as e:
        print(f"Error configuring Generative AI: {e}")
        return None

def analyze_statement_with_gemini(model, statement: str):
    """Analyzes a statement for misinformation using Gemini."""
    try:
        prompt = f"""
        Analyze the following statement for potential misinformation.
        Statement: "{statement}"
        Provide your analysis in a JSON format with these keys:
        - "classification": (e.g., "True", "False", "Misleading", "Unverifiable")
        - "confidence_score": (an integer from 0 to 100)
        - "explanation": (a brief, clear explanation for your classification)
        """
        response = model.generate_content(prompt)
        if not response or not response.text:
            raise ValueError("Received empty response from the AI service")

        # Clean the response to extract the JSON part
        json_response_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_response_text)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error processing AI response: {e}")
        return {
            "classification": "Analysis Failed",
            "confidence_score": 0,
            "explanation": "The AI response was not in the expected format. Please try again."
        }
    except Exception as e:
        print(f"Error in analyze_statement_with_gemini: {e}")
        return None

def fetch_fact_check_references(api_key: str, query: str):
    """Fetches fact-check articles from the Google Fact Check Tools API."""
    if not api_key:
        return None
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": query, "key": api_key, "languageCode": "en-US"}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
        if data.get('claims') and len(data['claims']) > 0:
            return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching fact-check data: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding fact-check API response: {e}")
        return None
    return None


def analyze_image(model, image_bytes: bytes):
    """Analyzes an image for potential misinformation using Gemini's vision capabilities."""
    try:
        image = PIL.Image.open(io.BytesIO(image_bytes))
        image_prompt = [
            """
            Analyze this image for potential misinformation. Look for:
            1. Signs of digital manipulation or editing
            2. Misleading captions or context
            3. Outdated images presented as recent
            4. Cropped or altered content
            Provide a brief analysis of what you see and any red flags for misinformation.
            """, image
        ]
        response = model.generate_content(image_prompt)
        if not response or not response.text:
            raise ValueError("Received empty response from the AI service")
        return response.text
    except Exception as e:
        print(f"Error in analyze_image: {e}")
        return None


# ===== ENHANCED CAPABILITIES WITH CORE LIBRARIES =====

class TruthLensDatabase:
    """SQLite database for storing fact-checking data and analysis history."""
    
    def __init__(self, db_path: str = "truthlens.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create claims table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_text TEXT NOT NULL,
                claim_hash TEXT UNIQUE NOT NULL,
                classification TEXT,
                confidence_score INTEGER,
                explanation TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                source_url TEXT,
                user_feedback TEXT
            )
        ''')
        
        # Create fact_sources table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fact_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id INTEGER,
                source_name TEXT,
                source_url TEXT,
                rating TEXT,
                title TEXT,
                FOREIGN KEY (claim_id) REFERENCES claims (id)
            )
        ''')
        
        # Create embeddings table for similarity search
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id INTEGER,
                embedding BLOB,
                model_name TEXT,
                FOREIGN KEY (claim_id) REFERENCES claims (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    def store_claim_analysis(self, claim_text: str, classification: str, 
                           confidence_score: int, explanation: str, 
                           source_url: Optional[str] = None, fact_sources: list = []):
        """Store a claim analysis in the database."""
        conn = sqlite3.connect(self.db_path)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create hash for the claim
        claim_hash = hashlib.md5(claim_text.encode()).hexdigest()
        
        try:
            # Insert claim
            cursor.execute('''
                INSERT OR REPLACE INTO claims 
                (claim_text, claim_hash, classification, confidence_score, explanation, source_url)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (claim_text, claim_hash, classification, confidence_score, explanation, source_url))
            
            claim_id = cursor.lastrowid
            
            # Insert fact sources if provided
            if fact_sources:
                for source in fact_sources:
                    cursor.execute('''
                        INSERT INTO fact_sources (claim_id, source_name, source_url, rating, title)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (claim_id, source.get('name', ''), source.get('url', ''), 
                          source.get('rating', ''), source.get('title', '')))
            
            conn.commit()
            return claim_id
            
        except Exception as e:
            print(f"Error storing claim analysis: {e}")
            return None
        finally:
            conn.close()
    
    def get_similar_claims(self, claim_text: str, limit: int = 5):
        """Find similar claims in the database using embeddings."""
        # This would be implemented with vector similarity search
        # For now, return basic text similarity
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                SELECT claim_text, classification, confidence_score, explanation, timestamp
                FROM claims 
                WHERE claim_text LIKE ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (f'%{claim_text[:50]}%', limit))
            
            results = cursor.fetchall()
            
            return [{
                'claim_text': row[0],
                'classification': row[1],
                'confidence_score': row[2],
                'explanation': row[3],
                'timestamp': row[4]
            } for row in results]
        except sqlite3.OperationalError:
            # This can happen if the 'claims' table doesn't exist yet.
            return []
        finally:
            conn.close()


class TruthLensEmbeddings:
    """Enhanced text embeddings using SentenceTransformers and FastEmbed."""
    
    def __init__(self):
        self.sentence_model = None
        self.fastembed_model = None
        self.init_models()
    
    def init_models(self):
        """Initialize embedding models with progress tracking."""
        try:
            print("🔄 Loading AI models... This may take a moment on first run.")
            
            # Initialize SentenceTransformers model (faster, lighter)
            print("📥 Loading SentenceTransformers model...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ SentenceTransformers model loaded successfully")
            
            # Skip FastEmbed for now to speed up loading
            # FastEmbed can be loaded on-demand if needed
            self.fastembed_model = None
            print("⚡ FastEmbed model skipped for faster startup (can be loaded on-demand)")
            
        except Exception as e:
            print(f"Error initializing embedding models: {e}")
            # Fallback: continue without embeddings
            self.sentence_model = None
            self.fastembed_model = None
    
    def get_sentence_embedding(self, text: str):
        """Get embedding using SentenceTransformers."""
        if not self.sentence_model:
            return None
        
        try:
            embedding = self.sentence_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"Error getting sentence embedding: {e}")
            return None
    
    def get_fastembed_embedding(self, text: str):
        """Get embedding using FastEmbed."""
        if not self.fastembed_model:
            return None
        
        try:
            # FastEmbed returns an iterator, so we need to convert it
            embeddings = list(self.fastembed_model.embed([text]))
            return embeddings[0].tolist() if embeddings else None
        except Exception as e:
            print(f"Error getting FastEmbed embedding: {e}")
            return None
    
    def calculate_similarity(self, text1: str, text2: str):
        """Calculate similarity between two texts using embeddings."""
        embedding1 = self.get_sentence_embedding(text1)
        embedding2 = self.get_sentence_embedding(text2)
        
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        return float(similarity)


class TruthLensAnalytics:
    """Data analysis and insights using Pandas and NumPy."""
    
    def __init__(self, db_path: str = "truthlens.db"):
        self.db_path = db_path
    
    def get_analysis_statistics(self):
        """Get comprehensive statistics about fact-checking analyses."""
        conn = sqlite3.connect(self.db_path)
        
        # Load data into pandas
        claims_df = pd.read_sql_query("SELECT * FROM claims", conn)
        sources_df = pd.read_sql_query("SELECT * FROM fact_sources", conn)
        
        conn.close()
        
        if claims_df.empty:
            return {
                'total_claims': 0,
                'accuracy_distribution': {},
                'confidence_stats': {},
                'top_sources': [],
                'trends': {}
            }
        
        # Calculate statistics
        stats = {
            'total_claims': len(claims_df),
            'accuracy_distribution': claims_df['classification'].value_counts().to_dict(),
            'confidence_stats': {
                'mean': float(claims_df['confidence_score'].mean()),
                'std': float(claims_df['confidence_score'].std()),
                'min': int(claims_df['confidence_score'].min()),
                'max': int(claims_df['confidence_score'].max())
            },
            'top_sources': sources_df['source_name'].value_counts().head(10).to_dict() if not sources_df.empty else {},
            'trends': {
                'daily_claims': (
                    claims_df.groupby(pd.to_datetime(claims_df['timestamp']).dt.date)
                    .size().astype(str).to_dict() if not claims_df.empty else {}
                )
            }
        }
        
        return stats
    
    def get_confidence_analysis(self):
        """Analyze confidence score patterns."""
        conn = sqlite3.connect(self.db_path)
        claims_df = pd.read_sql_query("SELECT * FROM claims", conn)
        conn.close()
        
        if claims_df.empty:
            return {}
        
        # Calculate confidence statistics
        confidence_stats = {
            'mean_confidence': float(claims_df['confidence_score'].mean()),
            'median_confidence': float(claims_df['confidence_score'].median()),
            'high_confidence_claims': int((claims_df['confidence_score'] >= 80).sum()),
            'low_confidence_claims': int((claims_df['confidence_score'] < 50).sum()),
            'confidence_by_classification': claims_df.groupby('classification')['confidence_score'].mean().to_dict()
        }
        
        return confidence_stats


def enhanced_analyze_statement(model, statement: str, db: Optional[TruthLensDatabase] = None, 
                             embeddings: Optional[TruthLensEmbeddings] = None):
    """Enhanced statement analysis with database storage and similarity search."""
    
    # First, check for similar claims in database
    if db:
        similar_claims = db.get_similar_claims(statement)
        if similar_claims:
            print(f"Found {len(similar_claims)} similar claims in database")
    
    # Perform original Gemini analysis
    analysis = analyze_statement_with_gemini(model, statement)
    
    if analysis and db:
        # Store the analysis in database
        claim_id = db.store_claim_analysis(
            claim_text=statement,
            classification=analysis.get('classification', 'Unknown'),
            confidence_score=analysis.get('confidence_score', 0),
            explanation=analysis.get('explanation', 'No explanation provided')
        )
        
        if claim_id:
            print(f"Analysis stored in database with ID: {claim_id}")
    
    return analysis


def get_truthlens_insights(db_path: str = "truthlens.db"):
    """Get comprehensive insights about TruthLens usage and performance."""
    analytics = TruthLensAnalytics(db_path)
    
    insights = {
        'statistics': analytics.get_analysis_statistics(),
        'confidence_analysis': analytics.get_confidence_analysis(),
        'recommendations': []
    }
    
    # Generate recommendations based on data
    stats = insights['statistics']
    if stats['total_claims'] > 0:
        if stats['confidence_stats']['mean'] < 70:
            insights['recommendations'].append("Consider improving fact-checking sources for higher confidence scores")
        
        if 'False' in stats['accuracy_distribution'] and stats['accuracy_distribution']['False'] > stats['total_claims'] * 0.3:
            insights['recommendations'].append("High percentage of false claims detected - consider enhancing detection algorithms")
    
    return insights

@st.cache_resource
def get_truthlens_db():
    """Returns a cached instance of TruthLensDatabase."""
    return TruthLensDatabase()

@st.cache_resource
def get_truthlens_embeddings():
    """Returns a cached instance of TruthLensEmbeddings."""
    return TruthLensEmbeddings()

@st.cache_resource
def get_truthlens_analytics():
    """Returns a cached instance of TruthLensAnalytics."""
    return TruthLensAnalytics()
