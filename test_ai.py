import json
from openai import OpenAI  # type: ignore

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def fix_transcript_typos(raw_transcript):
    system_prompt = (
        "You are a strict clinical spelling and typo correction utility. Your only job is to fix literal spelling mistakes, "
        "run-on words, and microphone phonetic errors (e.g., convert 'coff' to 'cough', or 'short breath/breaths' to 'shortness of breath').\n\n"
        "CRITICAL BOUNDARIES:\n"
        "1. STRICT LOWERCASE CASING: Retain standard sentence capitalization (capitalize the first letter of sentences and proper nouns). "
        "Do NOT randomly capitalize letters within a word or at the end of a word (e.g., 'cough' must never be written as 'cougH').\n"
        "2. NO LINGUISTIC REWRITING: Do not alter helper verbs, pronouns, or negation structures. Keep conversational patterns exactly as spoken.\n"
        "3. Output ONLY the clean text string. No conversational notes, commentary, or explanation."
    )
    try:
        response = client.chat.completions.create(
            model="llama3",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_transcript}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def generate_hpi_prose_from_data(extracted_data):
    system_prompt = (
        "You are a strict clinical prose generator. Your only job is to convert structured JSON medical data into a professional History of Present Illness (HPI) paragraph.\n\n"
        "CRITICAL PROTOCOLS:\n"
        "1. Include EVERY single active symptom, severity, duration, and trigger provided in the active_symptoms payload.\n"
        "2. MANDATORY NEGATIONS: Look closely at the 'negated_symptoms' array. You MUST explicitly state every denied or absent symptom listed there at the very end of the paragraph (e.g., 'The patient denies any fever.'). Do NOT drop or ignore this list.\n"
        "3. Use ONLY the facts provided. Do NOT invent, assume, or extrapolate any clinical details.\n"
        "4. Output ONLY the finalized clinical paragraph without introductions, explanations, or notes."
    )
    try:
        response = client.chat.completions.create(
            model="llama3",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(extracted_data)}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating HPI prose: {e}"

def ai_double_check_gaps(extracted_data, cleaned_transcript):
    system_prompt = (
        "You are a data validation agent. Compare the input JSON against the provided transcript slice to ensure accuracy.\n\n"
        "TASKS:\n"
        "1. Do not delete active symptoms.\n"
        "2. Review the transcript text. If any listed active symptoms contain null modifier values (such as duration or trigger) that are clearly declared in the transcript text, populate them.\n"
        "3. Keep formatting identical.\n\n"
        "Output raw JSON data directly. No markdown text formatting wrapping blocks."
    )
    user_payload = {
        "current_json": extracted_data,
        "transcript": cleaned_transcript
    }
    try:
        response = client.chat.completions.create(
            model="llama3",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)}
            ],
            temperature=0.0
        )
        clean_content = response.choices[0].message.content.strip()
        if clean_content.startswith("```json"):
            clean_content = clean_content.split("```json")[1].split("```")[0].strip()
        elif clean_content.startswith("```"):
            clean_content = clean_content.split("```")[1].split("```")[0].strip()
        return json.loads(clean_content)
    except Exception:
        return extracted_data