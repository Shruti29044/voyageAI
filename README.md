# voyageAI

# 🌍 VoyageAI: AI-powered Travel Planner (Multimodal RAG)

An intelligent travel itinerary generator that uses LLMs, image recognition, and real-time weather data to create personalized, 1-day travel plans — all wrapped in a friendly Gradio UI.

---

## 🚀 Features

- 🧠 **LLM-powered itinerary generation** using `LaMini-Flan-T5-783M`
- 🖼️ Optional **image analysis** with ResNet50 for landmark relevance
- 🧾 **Contextual sentiment filtering** from travel reviews via VADER
- ☀️ **Live weather integration** for realistic activity suggestions
- 🗂️ **ChromaDB + FAISS** for lightweight semantic search
- 💬 **Gradio UI** for easy, interactive user input
- ✅ Fully Colab-compatible

---

## 🛠️ Setup (Google Colab)

Install dependencies:
```bash
!pip install sentence-transformers faiss-cpu requests torch torchvision pillow transformers chromadb nltk gradio

(Optional) Upload an image for embedding via analyze_image_colab('/path/to/image.jpg')

Set your OpenWeather API key in the code:

pythonapi_key = "YOUR_WEATHER_API_KEY"


How It Works
Users enter a city and a trip preference (e.g. "historic sites and peaceful views").

The system embeds the text query and finds matching documents from ChromaDB.

It filters relevant content using sentiment analysis.

Weather info is pulled live via OpenWeather API.

The final prompt is sent to a local LLM via Hugging Face's transformers to generate a bullet-point itinerary.

Sample Prompt

City: Kyoto  
What kind of trip are you looking for?  
I enjoy local food, quiet cultural landmarks, and scenic spots for photography.

 Project Structure
generate_itinerary_colab(prompt, city) – generates the trip plan

get_weather(city) – fetches real-time weather

filter_positive_reviews() – filters positive sentiment docs

Gradio UI – frontend interface for real-time usage

Credits
Built with 🤖 Hugging Face Transformers, 🧠 Sentence Transformers, 🌦️ OpenWeather API, 🖼️ TorchVision, and Gradio.


Notes
This app uses local models only — no API keys needed for LLMs.

For best results, run with GPU enabled in Colab.





