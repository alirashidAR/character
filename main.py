import json
import os
import random
from collections import Counter
from typing import List
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Form
import requests
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId

# Load environment variables
load_dotenv()

# Load data
data = pd.read_json("clustered_characters.json")
with open("cluster_archetypes.json") as f:
    cluster_archetypes = json.load(f)
full_character_info = pd.read_json("character_codex.json")

# MongoDB setup
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    raise RuntimeError("MONGO_URI is not set in environment variables")
client = MongoClient(mongo_uri)
db = client["test"]
users_collection = db["users"]

# FastAPI setup
app = FastAPI(title="Character Archetype API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function to clean MongoDB documents
def clean_object_ids(obj):
    if isinstance(obj, dict):
        return {k: clean_object_ids(v) for k, v in obj.items() if k != "_id"}
    elif isinstance(obj, list):
        return [clean_object_ids(item) for item in obj]
    elif isinstance(obj, ObjectId):
        return str(obj)
    return obj

# Pydantic model
class UserSelection(BaseModel):
    character_names: List[str]

@app.get("/")
def root():
    return {"message": "Character Archetype API running"}

@app.post("/characters/full_info")
def get_full_character_info(selection: UserSelection):
    selection_lower = [name.lower() for name in selection.character_names]
    temp = full_character_info.copy()
    temp['character_name_lower'] = temp['character_name'].str.lower()

    matched = temp[temp['character_name_lower'].isin(selection_lower)]

    if matched.empty:
        return {"error": "No matching characters found in full character info"}

    results = matched[[
        'character_name', 'media_source', 'media_type', 'genre', 'description'
    ]].to_dict(orient='records')

    matched_names = matched['character_name_lower'].tolist()
    unmatched_names = list(set(selection_lower) - set(matched_names))

    return {
        "count": len(results),
        "characters": results,
        "unmatched_names": unmatched_names
    }

@app.get("/characters")
def get_characters():
    characters = data[['character_name', 'media_type', 'archetype_cluster']].sort_values(by="character_name")
    return {
        "count": len(characters),
        "characters": characters.to_dict(orient='records')
    }

@app.get("/characters/names")
def get_character_names():
    names = sorted(data['character_name'].unique().tolist())
    return {"character_names": names}

@app.get("/clusters/stats")
def get_cluster_stats():
    counts = data['archetype_cluster'].value_counts().sort_index().to_dict()
    return {"cluster_counts": counts}

@app.get("/clusters/archetypes")
def get_all_archetypes():
    return cluster_archetypes

@app.post("/user/archetype")
def get_user_archetype(selection: UserSelection):
    selected = data[data['character_name'].isin(selection.character_names)]
    if selected.empty:
        return {"error": "No matching characters found"}

    cluster_counts = Counter(selected['archetype_cluster'].tolist())
    dominant = cluster_counts.most_common(1)[0][0]
    return {
        "dominant_archetype_cluster": dominant,
        **cluster_archetypes.get(str(dominant), {})
    }

@app.get("/blend/{user_a}/{user_b}")
def blend_users(user_a: str, user_b: str):
    user1 = users_collection.find_one({"user_id": user_a})
    user2 = users_collection.find_one({"user_id": user_b})

    if not user1 or not user2:
        raise HTTPException(status_code=404, detail="One or both users not found")

    user1 = clean_object_ids(user1)
    user2 = clean_object_ids(user2)

    genres1 = set([g.lower() for g in user1.get("genres", [])])
    genres2 = set([g.lower() for g in user2.get("genres", [])])
    archetypes1 = {a["id"]: a for a in user1.get("archetypes", [])}
    archetypes2 = {a["id"]: a for a in user2.get("archetypes", [])}
    media1 = set(user1.get("media_sources", []))
    media2 = set(user2.get("media_sources", []))


    common_genres = sorted(genres1 & genres2)
    common_media = sorted(media1 & media2)
    common_archetypes = [
        archetypes1[aid] for aid in archetypes1 if aid in archetypes2
    ]


    if not common_genres:
        all_genres = list(genres1.union(genres2))
        random.shuffle(all_genres)
        common_genres = all_genres[:2] if len(all_genres) >= 2 else all_genres


    temp = full_character_info.copy()
    temp['genre'] = temp['genre'].apply(lambda g: [str(genre).lower() for genre in g] if isinstance(g, list) else [str(g).lower()])

    matched_movies = temp[temp['genre'].apply(lambda g_list: any(genre in g_list for genre in common_genres))]


    matched_movies['genre_str'] = matched_movies['genre'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))


    recommended_movies = matched_movies[['media_source', 'genre_str', 'media_type']].drop_duplicates()
    recommended_movies['media_type'] = recommended_movies['media_type'].str.lower()
    movies_only = recommended_movies[recommended_movies['media_type'] == 'movies']

    # Fallback: if no movies found, sample from all matched media
    if not movies_only.empty:
        selected = movies_only
    elif not recommended_movies.empty:
        selected = recommended_movies
    else:
        selected = pd.DataFrame()

    if not selected.empty:
        recommended_list = selected.rename(columns={'genre_str': 'genre'}).sample(n=min(5, len(selected))).to_dict(orient='records')
    else:
        recommended_list = []

    blend_name = f"{user_a} × {user_b} Blend"

    return {
        "blend_name": blend_name,
        "shared_genres": common_genres,
        "shared_archetypes": common_archetypes,
        "shared_media_sources": common_media,
        "recommended_movies": recommended_list,
        "note": "No matching movies found." if not recommended_list else None,
        "user_a": user_a,
        "user_b": user_b
    }


@app.get("/media/characters_grouped")
def get_characters_grouped_by_media_type():
    valid_types = {"Movies", "Television Shows", "Anime"}
    temp = full_character_info.copy()
    
    # Ensure consistent casing
    temp['media_type'] = temp['media_type'].str.title()

    # Filter by allowed media types
    filtered = temp[temp['media_type'].isin(valid_types)]

    result = {}
    for _, row in filtered.iterrows():
        media = row['media_source']
        character_entry = {
            "character_name": row['character_name'],
            "genre": row['genre'],
            "media_type": row['media_type'],
            "description": row['description']
        }
        result.setdefault(media, {"characters": []})["characters"].append(character_entry)

    return result

@app.get("/media/available")
def get_available_media():
    valid_types = {"Movies", "Television Shows", "Anime"}
    temp = full_character_info.copy()

    # Normalize media type casing
    temp['media_type'] = temp['media_type'].str.title()
    
    # Filter to valid types
    filtered = temp[temp['media_type'].isin(valid_types)]

    result = {}
    for media_type in valid_types:
        media_list = sorted(
            filtered[filtered['media_type'] == media_type]['media_source'].unique().tolist()
        )
        result[media_type] = media_list

    return result

@app.get("/user/{user_id}")
def get_user_info(user_id: str):
    user = users_collection.find_one({"user_id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user = clean_object_ids(user)

    # Ensure genres and media sources are lists
    user['genres'] = user.get('genres', [])
    user['media_sources'] = user.get('media_sources', [])
    
    # Ensure archetypes is a list of dicts
    user['archetypes'] = user.get('archetypes', [])

    return {
        "user_id": user_id,
        "genres": user['genres'],
        "media_sources": user['media_sources'],
        "archetypes": user['archetypes']
    }


# Load OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not set in .env")

# Load local character catalog
try:
    with open("clustered_characters.json") as f:
        LOCAL_SHOW_CATALOG = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load clustered_characters.json: {e}")

# Utility to determine time of day
def get_time_of_day():
    h = datetime.now().hour
    if 5 <= h < 12: return "morning"
    if 12 <= h < 17: return "afternoon"
    if 17 <= h < 21: return "evening"
    return "night"

# Select candidate shows based on genre overlap
def get_candidate_shows(user_profile: dict, full_catalog: list, max_candidates: int = 50) -> list:
    liked_genres = user_profile.get("genres", [])
    if not liked_genres:
        return random.sample(full_catalog, min(len(full_catalog), max_candidates))
    
    liked_genres_set = {genre.lower() for genre in liked_genres}
    candidates = [
        s for s in full_catalog 
        if s.get("genre") and any(lg in s["genre"].lower() for lg in liked_genres_set)
    ]
    if len(candidates) > max_candidates:
        return random.sample(candidates, max_candidates)
    return candidates or random.sample(full_catalog, min(len(full_catalog), max_candidates))

# Generate messages for OpenRouter prompt
def create_openai_compatible_messages(user_profile: dict, mood: str, time_of_day: str, candidate_shows: list) -> list:
    system_prompt = """
You are a highly perceptive and creative "Daylist" curator for an Amazon Prime-style service. Your goal is to create a small, perfect, and compelling playlist for a user based on their unique taste and current mood.

Your Tasks:
1. Create a Catchy Title: Generate a unique, fun, and creative title for this personalized playlist.
2. Curate a List: From the provided "Candidate Show Catalog", select 5 to 7 shows that are the absolute best fit.
3. Write Compelling Scenarios: For each show you select, write a short, one-line scenario that explains *why* it's the perfect choice for them right now.
4. Respond ONLY with a single, valid JSON object. Do not add any text before or after the JSON. The required format is:
{"title": "...", "shows": [{"character_name": "...", "media_source": "...", "genre": "...", "scenario": "..."}, ...]}
"""
    liked_genres = ", ".join(user_profile.get("genres", ["not specified"]))
    liked_shows = ", ".join(user_profile.get("media_sources", ["not specified"]))
    catalog_text = "\n".join(
        f"- {s.get('character_name', 'N/A')} from '{s.get('media_source', 'N/A')}' (Genre: {s.get('genre', 'N/A')})"
        for s in candidate_shows
    )
    user_prompt = f"""
Here is the user and catalog information. Please generate the curated daylist now.

**User's Profile:**
- **Vibe/Genres:** {liked_genres}
- **Liked Shows:** {liked_shows}
- **Current Mood:** {mood}
- **Time of Day:** {time_of_day}

**Candidate Show Catalog (Only pick from this list):**
{catalog_text}
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

@app.post("/daylist")
def generate_daylist(user_id: str = Form(...), mood: str = Form(...)):
    user_doc = users_collection.find_one({"user_id": user_id})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")

    user_profile = clean_object_ids(user_doc)
    time_of_day = get_time_of_day()
    candidate_shows = get_candidate_shows(user_profile, LOCAL_SHOW_CATALOG)

    messages = create_openai_compatible_messages(user_profile, mood, time_of_day, candidate_shows)

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://daylist.yourapp.com",
                "X-Title": "Daylist Curation App",
            },
            json={
                "model": "mistralai/mistral-7b-instruct:free",
                "messages": messages,
                "response_format": {"type": "json_object"}
            }
        )
        response.raise_for_status()
        ai_response_json = response.json()
        raw_response_text = ai_response_json['choices'][0]['message']['content']
        result = json.loads(raw_response_text)
        return result
    except requests.exceptions.RequestException as e:
        if e.response is not None:
            raise HTTPException(status_code=500, detail=f"OpenRouter Error: {e.response.text}")
        raise HTTPException(status_code=500, detail="Failed to contact OpenRouter.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", port=5000)
