## 📘 Character Archetype API – Endpoints Documentation

### 🔍 `GET /`

Returns a basic health check message.

**Response:**

```json
{ "message": "Character Archetype API running" }
```

---

### 📚 `GET /characters`

Returns all available characters with their name, media type, and cluster ID.

**Response:**

```json
{
  "count": 100,
  "characters": [
    {
      "character_name": "Naruto Uzumaki",
      "media_type": "Anime",
      "archetype_cluster": 2
    },
    ...
  ]
}
```

---

### 🧾 `GET /characters/names`

Returns a sorted list of all unique character names.

**Response:**

```json
{
  "character_names": [
    "Ash Ketchum",
    "Diana Prince",
    "Naruto Uzumaki",
    ...
  ]
}
```

---

### 📊 `GET /clusters/stats`

Returns the number of characters in each cluster.

**Response:**

```json
{
  "cluster_counts": {
    "0": 22,
    "1": 18,
    "2": 20,
    "3": 25,
    "4": 15
  }
}
```

---

### 🧠 `GET /clusters/archetypes`

Returns detailed archetype information for each cluster.

**Response:**

```json
{
  "0": {
    "archetype_name": "Lone Strategist",
    "description": "Cunning and calculated characters who work in isolation.",
    "keywords": ["strategy", "leader", "cold", ...],
    "sample_characters": ["Light Yagami", "Frank Underwood", "Jigsaw"]
  },
  ...
}
```

---

### 🧍‍♂️ `POST /user/archetype`

Input a list of known character names and get the **dominant archetype** cluster among them.

**Request Body:**

```json
{
  "character_names": ["Naruto Uzumaki", "Ash Ketchum", "Satoru Gojo"]
}
```

**Response:**

```json
{
  "dominant_archetype_cluster": 2,
  "archetype_name": "Skilled Maverick",
  "description": "A highly skilled and combat-proficient character known for their unique demeanor and abilities within their series.",
  "keywords": ["skill", "battle", "resolve", ...],
  "sample_characters": ["Satoru Gojo", "Levi Ackerman", "Itachi Uchiha"]
}
```
