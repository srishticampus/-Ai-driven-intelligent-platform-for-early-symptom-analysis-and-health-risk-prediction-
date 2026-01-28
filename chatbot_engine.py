import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MedicalChatbot:
    def __init__(self):
        # Knowledge base: (Question, Answer, Disease Category)
        self.knowledge_base = [
            
            # Hepatitis
            ("What is hepatitis?", "Hepatitis is an inflammation of the liver. The condition can be self-limiting or can progress to fibrosis, cirrhosis, or liver cancer.", "Hepatitis"),
            ("Symptoms of hepatitis", "Symptoms include yellowing of the skin and eyes (jaundice), dark urine, extreme fatigue, nausea, vomiting, and abdominal pain.", "Hepatitis"),
            ("Types of hepatitis", "There are 5 main types: A, B, C, D, and E. B and C are the most common causes of liver cirrhosis and cancer.", "Hepatitis"),
            ("How is hepatitis transmitted?", "It depends on the type. Hepatitis A and E are usually spread through contaminated food or water. B, C, and D are spread through contact with infected body fluids.", "Hepatitis"),
            ("Hepatitis B vaccine", "There is a safe and effective vaccine that offers 98% to 100% protection against hepatitis B.", "Hepatitis"),
            ("Hepatitis C treatment", "Hepatitis C is now curable with highly effective antiviral medications.", "Hepatitis"),
            
            # Kidney Disease
            ("What is Chronic Kidney Disease (CKD)?", "CKD is a condition where the kidneys are damaged and cannot filter blood as well as they should. This causes waste to build up in the body.", "Kidney"),
            ("Symptoms of kidney disease", "Early stages may have no symptoms. Later symptoms include fatigue, swollen ankles, shortness of breath, blood in urine, and more frequent nighttime urination.", "Kidney"),
            ("Risk factors for kidney disease", "Major risk factors include diabetes, high blood pressure, heart disease, and a family history of kidney failure.", "Kidney"),
            ("How to keep kidneys healthy?", "Stay hydrated, eat a balanced diet, control your blood pressure and blood sugar, and avoid excessive use of over-the-counter painkillers.", "Kidney"),
            ("Creatinine levels", "Creatinine is a waste product that kidneys filter out. High levels in the blood may indicate impaired kidney function.", "Kidney"),
            
            # Parkinson's
            ("What is Parkinson's disease?", "Parkinson's is a progressive nervous system disorder that affects movement. It develops gradually, sometimes starting with a barely noticeable tremor.", "Parkinson's"),
            ("Early signs of Parkinson's", "Early signs include tremors, slowed movement (bradykinesia), rigid muscles, impaired posture/balance, and changes in speech or writing.", "Parkinson's"),
            ("Causes of Parkinson's", "The exact cause is unknown, but it involves the breakdown or death of neurons that produce dopamine in the brain.", "Parkinson's"),
            ("Management of Parkinson's", "While it can't be cured, medications, surgery (like DBS), and lifestyle modifications (physical therapy) can significantly improve symptoms.", "Parkinson's"),
            ("Tremors", "Tremors are involuntary, rhythmic muscle contractions that cause shaking. In Parkinson's, they often happen at rest.", "Parkinson's"),
            
            # General
            ("Hello", "Hello! I am your AI Health Assistant. I can help you with information about Hepatitis, Kidney Disease, and Parkinson's. What would you like to know?", "General"),
            ("Hi", "Hi there! I can provide educational information about various health conditions we monitor here. How can I assist you?", "General"),
            ("Who are you?", "I am a medical chatbot designed to assist with early symptom analysis and provide educational information about lifestyle diseases.", "General"),
            ("Who created you?", "I was developed to provide intelligent healthcare insights using machine learning.", "General"),
            ("What can you do?", "I can explain symptoms and prevention for Hepatitis, Kidney Disease, and Parkinson's. I also explain how our risk analysis works.", "General"),
            ("How does risk analysis work?", "Our system uses Machine Learning models trained on clinical datasets to predict the probability of a disease based on your health markers.", "General"),
            ("Thank you", "You're welcome! I'm here to help. Stay healthy!", "General")
        ]
        
        self.questions = [item[0] for item in self.knowledge_base]
        self.answers = [item[1] for item in self.knowledge_base]
        
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)

    def get_response(self, user_query):
        query_lower = user_query.lower().strip()
        
        # Priority 1: Exact/Keyword Matching for General Intents
        if any(word in query_lower for word in ["who are you", "what is your name", "who are you?"]):
            return "I am a medical chatbot designed to assist with early symptom analysis and provide educational information about lifestyle diseases like Hepatitis, Kidney Disease, and Parkinson's."
        
        if any(word in query_lower for word in ["hello", "hi", "hey"]):
            return "Hello! I am your AI Health Assistant. I can help you with information about Hepatitis, Kidney Disease, and Parkinson's. What would you like to know?"

        if any(word in query_lower for word in ["what can you do", "help"]):
            return "I can explain symptoms and prevention for Hepatitis, Kidney Disease, and Parkinson's. I also explain how our risk analysis works based on clinical markers."

        # Priority 2: Semantic Similarity for medical queries
        query_vec = self.vectorizer.transform([user_query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)
        best_match_idx = np.argmax(similarities)
        max_similarity = similarities[0, best_match_idx]
        
        if max_similarity > 0.2: 
            return self.answers[best_match_idx]
        else:
            return "I'm sorry, I don't have enough information about that. I am specifically trained to discuss Hepatitis, Kidney Disease, and Parkinson's. Feel free to ask about their symptoms or prevention!"

# Singleton instance
chatbot = MedicalChatbot()
