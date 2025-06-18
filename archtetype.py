import pandas as pd
import re
import json
import openai
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from dotenv import load_dotenv
import os


load_dotenv()

# Load dataset
data = pd.read_json('dataset/characters/character_codex.json')
media_types = {'Anime', 'Movies', 'Manga'}
filtered = data[data['media_type'].isin(media_types)].copy()

# Clean and preprocess descriptions
def clean_description(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = text.split()
    return ' '.join([word for word in tokens if word not in ENGLISH_STOP_WORDS and len(word) > 2])

filtered['cleaned_description'] = filtered['description'].apply(clean_description)

# Vectorize
vectorizer = TfidfVectorizer(max_features=100)
X = vectorizer.fit_transform(filtered['cleaned_description'])

# Clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
filtered['archetype_cluster'] = kmeans.fit_predict(X)

# Save clustered data
filtered.to_json("clustered_characters.json", orient="records", indent=2)

# --- Archetype labeling ---

# Get top keywords and sample characters for each cluster
def get_top_keywords(cluster_id, n_words=10):
    feature_names = vectorizer.get_feature_names_out()
    cluster_center = kmeans.cluster_centers_[cluster_id]
    return [feature_names[i] for i in cluster_center.argsort()[::-1][:n_words]]

def get_sample_characters(cluster_id, n_samples=3):
    return filtered[filtered['archetype_cluster'] == cluster_id]['character_name'].head(n_samples).tolist()

# Fix improperly formatted JSON from OpenRouter
def clean_and_parse_openrouter_json(raw_content: str):
    cleaned = re.sub(r"^```json\n|```$", "", raw_content.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        return {
            "archetype_name": "Unknown",
            "description": "Could not parse response"
        }

# OpenRouter call
def get_archetype_name_openrouter(keywords, characters):
    api_key = os.getenv("OPENROUTER_API_KEY")

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    prompt = (
        "Given the following keywords and character names from a story cluster, "
        "suggest a creative, concise archetype name (1-3 words) and a one-line description.Dont' use words like manga, anime, or movie in the archetype name.\n\n"
        "Return as JSON with keys 'archetype_name' and 'description'.\n\n"
        f"Keywords: {', '.join(keywords)}\n"
        f"Characters: {', '.join(characters)}"
    )

    response = client.chat.completions.create(
        model="deepseek/deepseek-chat-v3-0324:free",
        messages=[
            {"role": "system", "content": "You are an expert in storytelling. Respond in JSON with 'archetype_name' and 'description'."},
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content
    print(f"LLM Raw Response:\n{content}\n")
    return clean_and_parse_openrouter_json(content)

# Build and save archetypes
archetypes = {}
for cid in range(n_clusters):
    keywords = get_top_keywords(cid)
    characters = get_sample_characters(cid)
    result = get_archetype_name_openrouter(keywords, characters)
    archetypes[str(cid)] = {
        "keywords": keywords,
        "sample_characters": characters,
        "archetype_name": result.get("archetype_name", "Unknown"),
        "description": result.get("description", "")
    }

# Save archetypes
with open("cluster_archetypes.json", "w") as f:
    json.dump(archetypes, f, indent=2)
