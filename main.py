import json
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer

bot_name = "TalkMaster"


# Load knowledge base
def load_knowledge_base(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Save knowledge base
def save_knowledge_base(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

# Find best match using TF-IDF
def find_best_match(user_input, questions):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(questions + [user_input])
    similarities = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1])
    best_match_index = similarities.argmax()
    if similarities[best_match_index] > 0.6:
        return questions[best_match_index]
    else:
        return None

# Get answer for question from knowledge base
def get_answer_for_question(question, knowledge_base):
    for q in knowledge_base["questions"]:
        if q["question"] == question:
            return q["answer"]

# Generate response using GPT-2
def generate_gpt2_response(user_input):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Main chat bot function
def chat_bot():
    knowledge_base = load_knowledge_base('knowledgebase.json')

    while True:
        user_input = input('You: ')
        if user_input.lower() == 'quit':
            # Save knowledge base before quitting
            save_knowledge_base('knowledgebase.json', knowledge_base)
            break
        
        # Tokenize user input
        user_tokens = word_tokenize(user_input)

        # Check if user wants to teach the bot a new response
        if user_input.lower() == 'teach':
            new_question = input("Enter the question: ")
            new_answer = input("Enter the answer: ")
            knowledge_base["questions"].append({"question": new_question, "answer": new_answer})
            save_knowledge_base('knowledgebase.json', knowledge_base)
            print(f'{bot_name}: Thank you! I learned a new response.')
            continue
        
        # Generate response using GPT-2
        response = generate_gpt2_response(user_input)
        print(f'{bot_name}: {response}')

if __name__ == '__main__':
    chat_bot()
