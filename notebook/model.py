import nltk
import numpy as np
import random
import string  # to process standard python strings

# Download necessary NLTK packages
nltk.download('punkt')
nltk.download('wordnet')

# Sample training data
CORPUS = """
Hello! How can I assist you today?
I am a chatbot created to help answer your questions.
What would you like to know?
Can you tell me about your hobbies?
I enjoy helping users and chatting with them.
Goodbye! Have a nice day.
"""

# Tokenizing the text
sent_tokens = nltk.sent_tokenize(CORPUS)
word_tokens = nltk.word_tokenize(CORPUS)

# Preprocessing - Lemmatizing
lemmer = nltk.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token.lower()) for token in tokens if token not in string.punctuation]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower()))

# Generating responses using cosine similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_input):
    sent_tokens.append(user_input)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if req_tfidf == 0:
        return "I am sorry, I don't understand that."
    else:
        return sent_tokens[idx]

# Main chatbot loop
def chatbot():
    print("Chatbot: Hi! Type 'bye' to exit.")
    while True:
        user_input = input("You: ").lower()
        if user_input == 'bye':
            print("Chatbot: Goodbye! Have a nice day.")
            break
        else:
            print("Chatbot:", response(user_input))

# Run the chatbot
if __name__ == "__main__":
    chatbot()
