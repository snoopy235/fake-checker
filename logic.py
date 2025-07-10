import os
import re
import json
from datetime import datetime
from statistics import mean
from typing import Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import openai
from openai import OpenAI

# .env laden
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openrouter_key = os.getenv("OPENROUTER_API_KEY")

# DeepSeek über OpenRouter
deepseek_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_key
)
deepseek_model = "deepseek/deepseek-r1-0528"


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9äöüßÄÖÜ\s]", "", text)
    return text.strip()


def contains_current_reference(text: str) -> bool:
    keywords = ["aktuell", "heute", "jetzt", "kürzlich", "neu", "zurzeit", "momentan"]
    future_years = [str(y) for y in range(2022, datetime.now().year + 2)]
    return any(k in text.lower() for k in keywords) or any(y in text for y in future_years)


def fetch_website_text(url: str) -> str:
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return "\n".join(p.get_text() for p in soup.find_all("p")).strip()[:2000]
    except Exception as e:
        return f"Fehler beim Laden der Website: {e}"


def build_prompt(text: str, article: str, date_string: str) -> str:
    return f"""
Du bist ein neutraler Faktenprüfer.
Bewerte folgende markierte Aussage auf ihren Wahrheitsgehalt – mit Stand vom {date_string} – unter Berücksichtigung des Artikels:

Artikeltext:
{article}

Markierter Satz:
{text}

Antwort im JSON-Format:
{{
  "traffic_light": "grün/gelb/rot",
  "confidence": "zwischen 0.0 und 1.0",
  "explanation": "kurze Begründung"
}}
"""


def check_with_gpt(prompt: str) -> dict:
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        print("GPT-3.5 Antwort:", content)
        return json.loads(content)
    except Exception as e:
        print("GPT-3.5 Fehler:", e)
        return {
            "traffic_light": "gelb",
            "confidence": 0.0,
            "explanation": f"Fehler bei GPT-3.5: {e}"
        }


def check_with_gpt_web(prompt: str) -> dict:
    try:
        response = openai.responses.create(
            input=prompt,
            model="gpt-4o",
            tools=[{"type": "web_search"}],
            instructions=f"Verwende NUR Websuche. Stand: {datetime.now().strftime('%d.%m.%Y')}"
        )

        for item in response.output:
            if hasattr(item, "content") and item.content:
                text = item.content[0].text.strip()
                print("GPT-4o Websuche Antwort (roh):", text)

                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    return json.loads(match.group())
                else:
                    return {
                        "traffic_light": "gelb",
                        "confidence": 0.0,
                        "explanation": f"GPT-4o Websuche konnte nicht geparst werden:\n{text}"
                    }

        raise ValueError("Keine gültige GPT-Antwort gefunden")

    except Exception as e:
        print("GPT-4o Websuche Fehler:", e)
        return {
            "traffic_light": "gelb",
            "confidence": 0.0,
            "explanation": f"Fehler bei GPT-4o Websuche: {e}"
        }


def check_with_deepseek(prompt: str) -> dict:
    try:
        response = deepseek_client.chat.completions.create(
            model=deepseek_model,
            messages=[{"role": "user", "content": prompt}],
            extra_headers={
                "HTTP-Referer": "https://yourdomain.com",
                "X-Title": "FakeChecker"
            }
        )

        if not hasattr(response, "choices") or not response.choices:
            return {
                "traffic_light": "gelb",
                "confidence": 0.0,
                "explanation": "DeepSeek hat keine Antwort geliefert"
            }

        content = response.choices[0].message.content.strip()
        print("DeepSeek Antwort:", content)
        return json.loads(content)

    except Exception as e:
        print("DeepSeek Fehler:", e)
        return {
            "traffic_light": "gelb",
            "confidence": 0.0,
            "explanation": f"DeepSeek Fehler: {e}"
        }


def combine_results(results: list[dict]) -> dict:
    valid = []
    for r in results:
        try:
            confidence = float(r["confidence"])
            if confidence > 0:
                r["confidence"] = confidence
                valid.append(r)
        except (ValueError, TypeError):
            continue

    if not valid:
        return {
            "traffic_light": "gelb",
            "confidence": 0.0,
            "explanation": "Keine verwertbare Antwort verfügbar."
        }

    confidences = [r["confidence"] for r in valid]
    avg_conf = round(mean(confidences), 2)

    lights = [r["traffic_light"] for r in valid]
    if all(l == "grün" for l in lights):
        traffic = "grün"
    elif all(l == "rot" for l in lights):
        traffic = "rot"
    else:
        traffic = "gelb"

    explanation = "\n---\n".join(r["explanation"] for r in valid)
    return {"traffic_light": traffic, "confidence": avg_conf, "explanation": explanation}


def check_text(text: str, url: Optional[str] = None, date: Optional[str] = None) -> dict:
    article_text = fetch_website_text(url) if url else ""
    date_string = datetime.fromisoformat(date).strftime("%d.%m.%Y") if date else datetime.now().strftime("%d.%m.%Y")
    prompt = build_prompt(text, article_text, date_string)

    use_web = contains_current_reference(text) or date
    print("Websuche aktiv:", use_web)

    gpt_result = check_with_gpt_web(prompt) if use_web else check_with_gpt(prompt)
    deepseek_result = check_with_deepseek(prompt)

    combined = combine_results([gpt_result, deepseek_result])
    combined["source_url"] = str(url) if url else None
    return combined
