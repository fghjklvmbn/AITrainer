def json_to_text(data: dict) -> str:
    text = []

    text.append(f"[세계관] {data.get('world', '')}")
    text.append(f"[장르] {data.get('genre', '')}")
    text.append(f"[줄거리] {data.get('plot', '')}")
    text.append("\n[등장인물]")
    
    for char in data.get("characters", []):
        name = char.get("character_name", "").strip()
        gender = char.get("gender", "")
        personality = char.get("personality", "")
        ability = char.get("ability", "")
        main = "(주인공)" if char.get("main_character", False) else ""
        text.append(f"- {name} ({gender}) {main}: {personality} / 능력: {ability}")

    text.append(f"\n[진행] {data.get('story_progression', '')}")
    tags = ", ".join(data.get("tags", []))
    text.append(f"[태그] {tags}")
    
    return "\n".join(text)