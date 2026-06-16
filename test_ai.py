import json
from openai import OpenAI  # type: ignore

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def fix_transcript_typos(raw_transcript):
    system_prompt = (
        "You are a strict clinical spelling and typo correction utility. Your only job is to fix literal spelling mistakes, "
        "run-on words, and microphone phonetic errors (e.g., convert 'coff' to 'cough').\n\n"
        "UNIVERSAL CORRECTION PROTOCOLS:\n"
        "1. PHONETIC CONTEXT MATCHING: If a word is mangled or unrecognized, analyze its syllable sounds alongside surrounding clinical indicators (e.g., 'weekly', 'injection', 'mg', 'blood sugar'). You MUST map the unknown word to a medication that matches BOTH the phonetic sound AND the clinical context. Never swap a phonetically distant word just because it is a more common drug (e.g., do not replace a word sounding like 'mocharo' with 'metformin' if the text says 'injection').\n"
        "2. PRESERVE UNKNOWN MEDICAL TERMS: If an unrecognized word cannot be reliably matched phonetically to a known clinical term, leave the misspelled word EXACTLY as it was transcribed. Do NOT guess generic English words (like converting 'Moujaro' to 'Mystery') or substitute different medical conditions.\n"
        "3. CONVERT WORD NUMBERS: Convert any spelled-out numbers into standard digits.\n"
        "4. Output ONLY the clean text string. No conversational notes, commentary, or explanation."
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
        "You are a strict clinical prose generator. Your job is to format the provided clinical text stream into a clean, concise medical narrative.\n\n"
        "CRITICAL PROTOCOLS:\n"
        "1. PRESERVE SHORTHAND: Do NOT expand medical abbreviations or symbols back into full words. You MUST retain tokens like 'Pt', 'f/u', 'DM2', 'c/o', 'UTD', and '2x' exactly as they appear in the data.\n"
        "2. CONCISE STYLE: Write in short, direct clinical sentences or distinct fragmented blocks. Avoid conversational transitions or fluff.\n"
        "3. STRICTLY NO INTRODUCTIONS/OUTROS: Do NOT include meta-text, conversational introductions, or commentary. Do NOT write phrases like 'Here is the formatted clinical text:', 'Here is the text:', or 'Based on the provided data:'. Start immediately with the medical narrative text.\n"
        "4. Use ONLY the facts provided. Do NOT invent, assume, or extrapolate any clinical details.\n"
        "5. Output ONLY the finalized clinical text without introductions, explanations, or notes."
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