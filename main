# VoyageAI: AI-powered Travel Planner (Multimodal RAG) - Google Colab Version

# ----------------------
# SETUP & DEPENDENCIES (Google Colab compatible)
# ----------------------
# Install necessary packages in Colab:
# !pip install sentence-transformers faiss-cpu requests torch torchvision pillow transformers chromadb nltk gradio

import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from sentence_transformers import SentenceTransformer
import faiss
import requests
from PIL import Image
import io
import chromadb
import json
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import gradio as gr

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load models
text_model = SentenceTransformer('all-MiniLM-L6-v2')
image_model = resnet50(weights=ResNet50_Weights.DEFAULT)
image_model.eval()

# Setup FAISS index
d = 384
index = faiss.IndexFlatL2(d)

# Setup ChromaDB (stub for now)
client = chromadb.Client()
collection = client.get_or_create_collection(name="travel_data")

# Load Hugging Face LLM
llm_pipeline = pipeline("text-generation", model="MBZUAI/LaMini-Flan-T5-783M", device_map="auto", max_new_tokens=512)

# Preprocess image
def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)

# Get weather
def get_weather(city):
    api_key = "YOUR_WEATHER_API_KEY"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    response = requests.get(url)
    try:
        weather_data = response.json()
        description = weather_data.get('weather', [{}])[0].get('description', 'unavailable')
        return description
    except Exception as e:
        return "unavailable"

# Analyze sentiment
def filter_positive_reviews(reviews):
    return [review for review in reviews if sia.polarity_scores(review)['compound'] > 0.2]

# Analyze image (Simulated upload in Colab)
def analyze_image_colab(image_path):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    input_tensor = preprocess_image(image_bytes)
    with torch.no_grad():
        features = image_model(input_tensor).squeeze().numpy()
    return features.tolist()[:10]

# Generate itinerary using Hugging Face LLM
def generate_itinerary_colab(prompt, city):
    text_embedding = text_model.encode(prompt).tolist()
    results = collection.query(query_embeddings=[text_embedding], n_results=5)
    documents = results["documents"][0] if results["documents"] else []
    filtered_docs = filter_positive_reviews(documents)
    context = "\n".join(filtered_docs)

    weather_desc = get_weather(city)
    user_message = f"itinerary for {city}."

    generated = llm_pipeline(user_message)[0]['generated_text']

    return generated

# Launch Gradio UI
def gradio_interface(city, prompt):
    result = generate_itinerary_colab(prompt, city)
    return result

gui = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="City"),
        gr.Textbox(label="What kind of trip are you looking for?", placeholder="e.g. I enjoy local food, quiet cultural landmarks, and scenic spots for photography. I prefer walking tours, avoid crowded tourist places, and love hidden gems with a historical story.")
    ],
    outputs="text",
    title="VoyageAI Travel Planner",
    description="Enter a city and your preferences to get a personalized travel itinerary."
)

gui.launch()

# Example usage (optional for backend testing):
# image_features = analyze_image_colab('/path/to/image.jpg')
# itinerary = generate_itinerary_colab('nature and cultural experiences', 'Kyoto')
# print(json.dumps(itinerary, indent=2))
