import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

# Sample training data (you can expand later)
corpus = [
    "Hello, how can I help you?",
    "What is your name?",
    "My name is KBot, I’m your assistant.",
    "How do I reset my password?"
    "You can set your password by clicking ‘Forgot Password’.",
    "Thanks!",
    "You’re welcome.",
    "Bye!",
    "Goodbye!"
]

vectorizer = CountVectorizer().fit_transform(corpus)
vectors = vectorizer.toarray()

def chatbot_response(user_input):
    corpus_with_input = corpus + [user_input]
    new_vectors = CountVectorizer().fit_transform(corpus_with_input).toarray()
    similarity = cosine_similarity([new_vectors[-1]], new_vectors[:-1])
    
    response_index = similarity.argmax()
    confidence = similarity[0][response_index]

    if confidence < 0.3:
        return "I'm not sure how to respond to that."
    return corpus[response_index]

# Main loop
print("🤖 KBot is online! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("KBot: Goodbye!")
        break
    print("KBot:", chatbot_response(user_input))
