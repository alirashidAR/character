import json
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from collections import Counter

# Load clustered characters and archetypes
data = pd.read_json("clustered_characters.json")
with open("cluster_archetypes.json") as f:
    cluster_archetypes = json.load(f)

# Load full character info JSON
full_character_info = pd.read_json("character_codex.json")  # This is your new JSON file


# Initialize FastAPI app
app = FastAPI(title="Character Archetype API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for user selection input
class UserSelection(BaseModel):
    character_names: List[str]

# Root endpoint
@app.get("/")
def root():
    return {"message": "Character Archetype API running"}

# POST endpoint to fetch media sources from full character info
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

# Get full character list
@app.get("/characters")
def get_characters():
    characters = data[['character_name', 'media_type', 'archetype_cluster']].sort_values(by="character_name")
    return {
        "count": len(characters),
        "characters": characters.to_dict(orient='records')
    }

# Get just the character names (for dropdowns or UIs)
@app.get("/characters/names")
def get_character_names():
    names = sorted(data['character_name'].unique().tolist())
    return {"character_names": names}

# Get cluster stats (number of characters in each cluster)
@app.get("/clusters/stats")
def get_cluster_stats():
    counts = data['archetype_cluster'].value_counts().sort_index().to_dict()
    return {"cluster_counts": counts}

# Get full archetype descriptions and sample characters
@app.get("/clusters/archetypes")
def get_all_archetypes():
    return cluster_archetypes

# Get dominant archetype based on selected character names
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

# Run with `python character_archetypes_api.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", port=5000 ,reload=True)
