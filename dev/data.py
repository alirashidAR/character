import pandas as pd
from pymongo import MongoClient
import json
import os
from dotenv import load_dotenv

load_dotenv("../.env")

# Load data
full_df = pd.read_json("../character_codex.json")
clustered_df = pd.read_json("../clustered_characters.json")
with open("../cluster_archetypes.json") as f:
    cluster_archetypes = json.load(f)

# Merge character info with archetype clusters
merged_df = pd.merge(full_df, clustered_df[['character_name', 'archetype_cluster']], on="character_name", how="left")

# MongoDB connection
client = MongoClient(os.getenv("MONGO_URI"))
db = client["character_blend"]
collection = db["users"]

# Sample users with favorite characters
user_favs = {
    "user_a": ["Hannibal Lecter", "Lady Bird McPherson", "Guts"],
    "user_b": ["Dr. Dakota Block", "The Dude (Jeffrey Lebowski)", "Mary Magdalene"]
}

def create_user_document(user_id, fav_names):
    characters = merged_df[merged_df["character_name"].isin(fav_names)]
    
    # Prepare enriched favorite character list
    enriched_chars = []
    archetype_ids = set()
    
    for _, row in characters.iterrows():
        if pd.isna(row['archetype_cluster']):
            continue  # or log this issue

        archetype_id = str(int(float(row['archetype_cluster'])))  # Convert '0.0' → '0'
        archetype_info = cluster_archetypes.get(archetype_id, {})
        archetype_ids.add(archetype_id)

        enriched_chars.append({
            "character_name": row["character_name"],
            "media_source": row["media_source"],
            "media_type": row["media_type"],
            "genre": row["genre"],
            "archetype_cluster": int(archetype_id),
            "archetype_name": archetype_info.get("archetype_name")
        })


    # Assemble archetype summaries
    enriched_archetypes = []
    for aid in archetype_ids:
        info = cluster_archetypes[aid]
        enriched_archetypes.append({
            "id": int(aid),
            "name": info["archetype_name"],
            "description": info["description"]
        })

    # Final user document
    return {
        "user_id": user_id,
        "favorite_characters": enriched_chars,
        "archetypes": enriched_archetypes,
        "genres": list(set(c["genre"] for c in enriched_chars)),
        "media_sources": list(set(c["media_source"] for c in enriched_chars))
    }

# Create and insert dummy users
collection.delete_many({})
for uid, favs in user_favs.items():
    user_doc = create_user_document(uid, favs)
    collection.insert_one(user_doc)
    print(f"Inserted {uid} with {len(user_doc['favorite_characters'])} characters.")
