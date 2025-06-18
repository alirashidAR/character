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
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
