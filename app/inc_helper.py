# app/inc_helper.py
import json
import random
from pathlib import Path
from datetime import datetime

INC_DIR = Path("inc_requests")
INC_DIR.mkdir(exist_ok=True)

def generate_ticket_number() -> str:
    """Generate a pseudo ServiceNow ticket number."""
    return f"INC{random.randint(10000000, 99999999)}"

def create_incident_payload(user_inputs: dict) -> dict:
    """Builds a standard ServiceNow incident creation payload."""
    return {
        "short_description": user_inputs.get("short_description", "N/A"),
        "description": user_inputs.get("long_description", "N/A"),
        "category": user_inputs.get("category", "General"),
        "impact": user_inputs.get("impact", "3 - Low"),
        "urgency": user_inputs.get("urgency", "3 - Low"),
        "caller": user_inputs.get("caller", "User"),
        "username": user_inputs.get("username", "username"),
        "user_email": user_inputs.get("contact_email", "contact_email"),
        "team": user_inputs.get("team_name", "N/A"),
        "application": user_inputs.get("application", "N/A"),
        "assigned_to": user_inputs.get("assigned_to", None),
        "opened_at": datetime.utcnow().isoformat() + "Z",
        "state": "New",
    }

def save_incident_json(payload: dict, ticket_number: str) -> Path:
    """Save payload as JSON under inc_requests/ folder."""
    file_path = INC_DIR / f"{ticket_number}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return file_path
