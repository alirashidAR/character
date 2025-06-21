import json
import os
import random
from collections import Counter
from typing import List

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Load data
data = pd.read_json("clustered_characters.json")
with open("cluster_archetypes.json") as f:
    cluster_archetypes = json.load(f)
full_character_info = pd.read_json("character_codex.json")

# MongoDB setup
client = MongoClient(os.getenv("MONGO_URI"))
db = client["character_blend"]
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

    genres1 = set(user1.get("genres", []))
    genres2 = set(user2.get("genres", []))
    archetypes1 = {a["id"]: a for a in user1.get("archetypes", [])}
    archetypes2 = {a["id"]: a for a in user2.get("archetypes", [])}
    media1 = set(user1.get("media_sources", []))
    media2 = set(user2.get("media_sources", []))

    # Find shared elements
    common_genres = sorted(genres1 & genres2)
    common_media = sorted(media1 & media2)
    common_archetypes = [
        archetypes1[aid] for aid in archetypes1 if aid in archetypes2
    ]

    # Fallback: pick 2 random genres from the union if no common genres
    if not common_genres:
        all_genres = list(genres1.union(genres2))
        random.shuffle(all_genres)
        common_genres = all_genres[:2] if len(all_genres) >= 2 else all_genres

    # Recommend movies from those genres
    temp = full_character_info.copy()
    temp['genre'] = temp['genre'].apply(lambda g: g if isinstance(g, list) else [g])
    matched_movies = temp[temp['genre'].apply(lambda g_list: any(genre in g_list for genre in common_genres))]

    # Fix unhashable type error
    matched_movies['genre_str'] = matched_movies['genre'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))

    # Filter and deduplicate
    recommended_movies = matched_movies[['media_source', 'genre_str', 'media_type']].drop_duplicates()
    recommended_movies = recommended_movies[recommended_movies['media_type'].str.lower() == 'movies']
    recommended_list = recommended_movies.rename(columns={'genre_str': 'genre'}).sample(n=min(5, len(recommended_movies))).to_dict(orient='records')

    blend_name = f"{user_a} × {user_b} Blend"

    return {
        "blend_name": blend_name,
        "shared_genres": common_genres,
        "shared_archetypes": common_archetypes,
        "shared_media_sources": common_media,
        "recommended_movies": recommended_list,
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app",host="0.0.0.0", port=5000)
