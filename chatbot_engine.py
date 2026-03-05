import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MedicalChatbot:
    def __init__(self):
        # 1. Hardcoded Intents (Priority General Questions)
        self.priority_intents = [
            ("Hello", "Hello! I am your AI Health Assistant. How can I help you today?", ["hi", "hello", "hey", "greetings"]),
            ("Who are you?", "I am a medical chatbot designed to assist with early symptom analysis and provide educational information from health datasets like MedQuAD.", ["who are you", "what is your name", "identity"]),
            ("What can you do?", "I can explain symptoms, causes, and treatments for thousands of diseases using the MedQuAD database, and I also assist in analyzing your health risk profiles from lab reports.", ["what can you do", "help", "features", "capabilities"]),
            ("How does risk analysis work?", "Our system uses Machine Learning models trained on clinical datasets to predict the probability of a disease based on your health markers extracted from reports.", ["risk analysis", "how it works", "prediction logic"]),
            ("Thank you", "You're welcome! I'm here to help. Stay healthy!", ["thanks", "thank you", "okay", "informative"])
        ]

        # 2. Load MedQuAD Knowledge Base
        self.questions = []
        self.answers = []
        
        csv_path = "medquad.csv"
        if os.path.exists(csv_path):
            try:
                print(f"Loading {csv_path}...")
                df = pd.read_csv(csv_path)
                # Ensure we have the right columns
                if 'question' in df.columns and 'answer' in df.columns:
                    # Clean the data: remove rows with empty questions or answers
                    df = df.dropna(subset=['question', 'answer'])
                    self.questions = df['question'].astype(str).tolist()
                    self.answers = df['answer'].astype(str).tolist()
                    print(f"Successfully loaded {len(self.questions)} QA pairs from MedQuAD.")
                else:
                    print(f"Warning: {csv_path} does not have required 'question' and 'answer' columns.")
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
        else:
            print(f"Warning: {csv_path} not found. Chatbot will use limited intents.")

        # 3. Add priority intents to the searchable pool (optional, kept for fallthrough)
        for q, a, _ in self.priority_intents:
            self.questions.append(q)
            self.answers.append(a)

        # 4. Initialize Vectorizer
        print("Vectorizing knowledge base... This may take a moment.")
        self.vectorizer = TfidfVectorizer(stop_words='english')
        if self.questions:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)
        else:
            self.tfidf_matrix = None
            print("Error: No questions found to vectorize.")

    def get_response(self, user_query):
        query_lower = user_query.lower().strip()
        
        # Priority 1: Check Hardcoded Priority Intents First (Exact keyword match)
        for intent_q, intent_a, keywords in self.priority_intents:
            if any(key in query_lower for key in keywords):
                return intent_a

        # Priority 2: Semantic Similarity for MedQuAD queries
        if self.tfidf_matrix is not None:
            try:
                query_vec = self.vectorizer.transform([user_query])
                similarities = cosine_similarity(query_vec, self.tfidf_matrix)
                best_match_idx = np.argmax(similarities)
                max_similarity = similarities[0, best_match_idx]
                
                # Using a higher threshold (0.3) for the large dataset to avoid irrelevant answers
                if max_similarity > 0.3: 
                    return self.answers[best_match_idx]
            except Exception as e:
                return f"I encountered an error while searching: {str(e)}"

        return "I'm sorry, I don't have enough information about that specific query. I am trained on general medical knowledge and the MedQuAD dataset. Please try asking about specific symptoms, treatments or causes of a condition."

# Singleton instance
chatbot = MedicalChatbot()
