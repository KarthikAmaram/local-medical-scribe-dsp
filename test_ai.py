import json
from openai import OpenAI  # type: ignore

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

FEW_SHOT_EXAMPLES = [
    {
        "text": "Pt is here for his cholesterol and diabetic check. His cardiologist had increased his cholesterol medication dosage. He was fasting for his appointment.",
        "topics": [
            {"topic": "Visit reason", "details": "Here for cholesterol and diabetic check."},
            {"topic": "Cholesterol", "details": "Cardiologist increased cholesterol medication dosage."},
            {"topic": "Fasting status", "details": "Was fasting for appointment."}
        ]
    },
    {
        "text": "Patient is here for check up. Patient is doing well. Last labs not available. UTD with all screenings and vaccines. Pt still has knee pain, did 8 weeks with chiro improved at first but pain returned. Monitors BP at home, highest 144/98, lowest 121/89. Pt also went to sleep doctor for sleep apnea, has machine but machine kept coming off during the night so Pt stopped treatment, does plan to try again.",
        "topics": [
            {"topic": "Visit reason", "details": "Here for check up."},
            {"topic": "General status", "details": "Doing well."},
            {"topic": "Labs", "details": "Last labs not available."},
            {"topic": "Screenings/vaccines", "details": "UTD with all screenings and vaccines."},
            {"topic": "Knee pain", "details": "Still has knee pain. Did 8 weeks of chiro, improved at first but pain returned."},
            {"topic": "Blood pressure", "details": "Monitors BP at home, highest 144/98, lowest 121/89."},
            {"topic": "Sleep apnea", "details": "Went to sleep doctor for sleep apnea, has machine but it kept coming off during the night, so Pt stopped treatment. Plans to try again."}
        ]
    },
    {
        "text": "Pt is here for f/u on DM2, is currently fasting. Pt reports monitoring fasting glucose 100-120 but only checks on the weekends. Has stopped Ozempic due to gastric issues.",
        "topics": [
            {"topic": "Visit reason", "details": "Here for f/u on DM2."},
            {"topic": "Fasting status", "details": "Currently fasting."},
            {"topic": "Glucose monitoring", "details": "Monitors fasting glucose 100-120, only checks on weekends."},
            {"topic": "Medications", "details": "Stopped Ozempic due to gastric issues."}
        ]
    },
    {
        "text": "Pt was here for her physicals and BP f/u. She was fasting for her appointment. Her diastolic blood pressure at home is a little high. She has stopped eating rice and has started rigorous weight training and exercising. She denied breast exam as she had recently done her mammogram. Her neuro has started her with headache shots as she was having bad headache/migraine for 1 week and after the shot she felt better and her antipsychotic medication is also changed. She is losing weight and planning to go to India in next few weeks. UTD with all screenings and vaccines. She wants to check her fasting insulin.",
        "topics": [
            {"topic": "Visit reason", "details": "Here for physical and BP f/u."},
            {"topic": "Fasting status", "details": "Fasting for appointment."},
            {"topic": "Blood pressure", "details": "Diastolic BP at home is a little high."},
            {"topic": "Diet/exercise", "details": "Stopped eating rice, started rigorous weight training and exercising."},
            {"topic": "Breast exam", "details": "Denied breast exam, recently had mammogram done."},
            {"topic": "Headaches/migraine", "details": "Neuro started headache shots for bad headache/migraine lasting 1 week, felt better after shot. Antipsychotic medication also changed."},
            {"topic": "Weight", "details": "Losing weight, planning trip to India in next few weeks."},
            {"topic": "Screenings/vaccines", "details": "UTD with all screenings and vaccines."},
            {"topic": "Labs requested", "details": "Wants to check fasting insulin."}
        ]
    },
    {
        "text": "Patient is here for physical and establish care for her diabetes. Pt moved from Chicago in Sep 2025. Her last labs and physical were done in Jan 2025 and the last labs for diabetic check were done in Jan 2026 in Chicago and recalls her hba1c about 7.1. Her iron level in Jan was 9.5. Her pap was also abnormal and when done the second time it was normal. She has chronic heartburn or stomach issues and was seen by GI in Chicago. She stated she has a gunshot wound to her abdomen which led to a small bowel resection, since then she has GI issues. Her sister has Crohn's disease. She also had a colonoscopy which did not show Crohn's. She has had an EGD and now uses PPI daily most of the time. She stated her FBS has always been >130. She could not tolerate a high dose of Trulicity and never tried Ozempic/Mounjaro. She had gained a lot of weight after coming to Dallas. She stated she doesn't eat much but her activity has been reduced at her current job. She c/o snoring but no sleep issues.",
        "topics": [
            {"topic": "Visit reason", "details": "Here for physical and to establish care for diabetes."},
            {"topic": "History", "details": "Moved from Chicago in Sep 2025."},
            {"topic": "Labs", "details": "Last labs and physical done Jan 2025. Last diabetic labs done Jan 2026 in Chicago, recalls hba1c about 7.1. Iron level in Jan was 9.5."},
            {"topic": "Pap smear", "details": "Pap was abnormal, repeat was normal."},
            {"topic": "GI history", "details": "Chronic heartburn/stomach issues, seen by GI in Chicago. Gunshot wound to abdomen led to small bowel resection, has had GI issues since. Colonoscopy did not show Crohn's. Had EGD, now uses PPI daily most of the time."},
            {"topic": "Family history", "details": "Sister has Crohn's disease."},
            {"topic": "Diabetes", "details": "FBS has always been >130. Could not tolerate high dose of Trulicity, never tried Ozempic/Mounjaro."},
            {"topic": "Weight", "details": "Gained a lot of weight after moving to Dallas. Doesn't eat much but activity reduced at current job."},
            {"topic": "Sleep", "details": "C/o snoring but no sleep issues."}
        ]
    },
    {
        "text": "Patient is here for his f/u. He was fasting and is continuing all his medications. He was also diagnosed with skin cancer on the right side of the face and also on the back. He has an f/u with his dermatologist.",
        "topics": [
            {"topic": "Visit reason", "details": "Here for f/u."},
            {"topic": "Fasting status", "details": "Fasting, continuing all medications."},
            {"topic": "Skin cancer", "details": "Diagnosed with skin cancer on right side of face and on back. Has f/u with dermatologist."}
        ]
    },
    {
        "text": "Pt is here for her f/u. She was fasting and continuing all her medications. She was diagnosed with Basal cell carcinoma after she found a pimple on the nose, after the procedure she again got a pimple on the left cheek and when checked it was benign. She has f/u appointment with her dermatologist.",
        "topics": [
            {"topic": "Visit reason", "details": "Here for f/u."},
            {"topic": "Fasting status", "details": "Fasting, continuing all medications."},
            {"topic": "Skin cancer", "details": "Diagnosed with basal cell carcinoma after finding pimple on nose. After procedure, got another pimple on left cheek, checked and was benign. Has f/u appointment with dermatologist."}
        ]
    },
    {
        "text": "Patient is here for immigration physical. Reviewed patient's paperwork. Patient denies any cough, night sweats or weight loss. Patient is not UTD with vaccines. Wife has Hodgkins lymphoma and did bone marrow transplant so he was advised not to take live vaccines.",
        "topics": [
            {"topic": "Visit reason", "details": "Here for immigration physical. Paperwork reviewed."},
            {"topic": "Respiratory/constitutional symptoms", "details": "Denies cough, night sweats, or weight loss."},
            {"topic": "Vaccines", "details": "Not UTD with vaccines."},
            {"topic": "Family history", "details": "Wife has Hodgkin's lymphoma and had bone marrow transplant, so Pt advised not to take live vaccines."}
        ]
    },
    {
        "text": "Pt still has headache, nausea, severe fatigue, joint pains, itchy all over his body especially hands and feet and fever. He had taken ibuprofen 40mg 2 hrs back without checking the temperature. He has been taking ibuprofen for the past 10 days on and off basis. He is also complaining of rash on his hands and arms. He is taking his antibiotics along with the other medication given in the urgent care.",
        "topics": [
            {"topic": "Symptoms", "details": "Still has headache, nausea, severe fatigue, joint pains, itchy all over body especially hands and feet, and fever."},
            {"topic": "Medications taken", "details": "Took ibuprofen 40mg 2 hours ago without checking temperature. Has been taking ibuprofen for past 10 days on and off."},
            {"topic": "Rash", "details": "Complaining of rash on hands and arms."},
            {"topic": "Current treatment", "details": "Taking antibiotics along with other medication given in urgent care."}
        ]
    },
    {
        "text": "Pt is here for f/u on DM2. Pt reports taking metformin 4x per day, only eats once per day. Has not checked fasting glucose since last visit.",
        "topics": [
            {"topic": "Visit reason", "details": "Here for f/u on DM2."},
            {"topic": "Medications", "details": "Taking metformin 4x per day."},
            {"topic": "Diet", "details": "Only eats once per day."},
            {"topic": "Glucose monitoring", "details": "Has not checked fasting glucose since last visit."}
        ]
    }
]

def fix_transcript_typos(raw_transcript):
    system_prompt = (
        "You are a strict clinical spelling and typo correction utility. Your only job is to fix literal spelling mistakes, "
        "run-on words, and microphone phonetic errors (e.g., convert 'coff' to 'cough').\n\n"
        "UNIVERSAL CORRECTION PROTOCOLS:\n"
        "1. PHONETIC CONTEXT MATCHING: If a word is mangled or unrecognized, analyze its syllable sounds alongside surrounding clinical indicators (e.g., 'weekly', 'injection', 'mg', 'blood sugar'). You MUST map the unknown word to a medication that matches BOTH the phonetic sound AND the clinical context. Never swap a phonetically distant word just because it is a more common drug (e.g., do not replace a word sounding like 'mocharo' with 'metformin' if the text says 'injection').\n"
        "2. PRESERVE UNKNOWN MEDICAL TERMS: If an unrecognized word cannot be reliably matched phonetically to a known clinical term, leave the misspelled word EXACTLY as it was transcribed. Do NOT guess generic English words (like converting 'Moujaro' to 'Mystery') or substitute different medical conditions.\n"
        "2b. KNOWN PHRASE-LEVEL PHONETIC CONFUSIONS: certain common clinical phrases are frequently mis-transcribed as a different but similarly-sounding phrase, most often 'up to date' being heard as 'up today'. If a sentence containing 'up today' is discussing screenings, vaccines, immunizations, or labs, correct it to 'up to date'. Only apply this correction when context clearly supports it; do not alter 'today' when used in its normal sense (e.g., 'patient is here today').\n"
        "3. CONVERT WORD NUMBERS: Convert any spelled-out numbers into standard digits.\n"
        "4. DO NOT REPHRASE: Do not reorder, rephrase, summarize, or remove any clinical content. Correct spelling and phonetic transcription errors only, word for word.\n"
        "5. Output ONLY the clean text string. No conversational notes, commentary, or explanation."
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

def extract_topics(cleaned_transcript):
    system_prompt = (
        "You are a clinical information extraction utility. Read a clinical transcript and break it into a list of "
        "topic/detail entries representing everything mentioned, including positive findings, negative findings, "
        "numeric values, doses, and frequencies.\n\n"
        "RULES:\n"
        "1. SPARSE OUTPUT: Only include topics that were actually mentioned. Never add a topic that was not discussed.\n"
        "2. PRESERVE NEGATIVES: If something was denied or stated as absent (e.g. 'no sleep issues', 'denies cough'), "
        "include it as its own detail entry, do not omit it.\n"
        "3. PRESERVE EXACT VALUES: Numbers, doses, frequencies, blood pressure readings, lab values, and dates must be "
        "copied exactly as written, never rounded or rephrased.\n"
        "4. PRESERVE SHORTHAND: Keep clinical shorthand tokens (Pt, f/u, DM2, c/o, UTD, etc.) exactly as they appear.\n"
        "5. DO NOT INVENT: Only use information explicitly present in the transcript. Do not infer or assume additional clinical detail.\n"
        "6. Each entry is an object with a 'topic' field (a short label for the subject) and a 'details' field "
        "(the relevant facts about that topic, concise, preserving exact values and shorthand).\n"
        "7. Output ONLY a JSON array of these objects. No markdown formatting, no commentary."
    )
    messages = [{"role": "system", "content": system_prompt}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["text"]})
        messages.append({"role": "assistant", "content": json.dumps(ex["topics"])})
    messages.append({"role": "user", "content": cleaned_transcript})

    try:
        response = client.chat.completions.create(
            model="llama3",
            messages=messages,
            temperature=0.0
        )
        clean_content = response.choices[0].message.content.strip()
        if clean_content.startswith("```json"):
            clean_content = clean_content.split("```json")[1].split("```")[0].strip()
        elif clean_content.startswith("```"):
            clean_content = clean_content.split("```")[1].split("```")[0].strip()
        return json.loads(clean_content)
    except Exception:
        return [{"topic": "Unparsed transcript", "details": cleaned_transcript}]

def generate_hpi_prose_from_data(extracted_data):
    system_prompt = (
        "You are a clinical note prose generator mimicking a specific physician's documentation style. You will be "
        "given a list of topic/detail entries from a single patient visit. Write them as a concise clinical narrative, "
        "in the style demonstrated in the examples.\n\n"
        "RULES:\n"
        "1. STYLE: Match the sentence style, brevity, and ordering conventions demonstrated in the examples. Each "
        "topic typically becomes one or a few short sentences.\n"
        "2. PRESERVE SHORTHAND: Do NOT expand shorthand tokens like Pt, f/u, DM2, c/o, UTD back into full words.\n"
        "3. PRESERVE EXACT VALUES: Numbers, doses, frequencies, and lab values must appear exactly as given, never "
        "rounded or altered.\n"
        "4. ONLY USE PROVIDED FACTS: Do NOT add, infer, or assume any clinical detail not present in the input data.\n"
        "5. NO PADDING: Do NOT mention topics that are not present in the input. Do NOT write filler sentences about "
        "missing information.\n"
        "6. STRICTLY NO INTRODUCTIONS/OUTROS: Start immediately with the clinical narrative. No meta-text, no phrases "
        "like 'Here is the note:'.\n"
        "7. Output ONLY the finalized clinical text."
    )
    messages = [{"role": "system", "content": system_prompt}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": json.dumps(ex["topics"])})
        messages.append({"role": "assistant", "content": ex["text"]})
    messages.append({"role": "user", "content": json.dumps(extracted_data)})

    try:
        response = client.chat.completions.create(
            model="llama3",
            messages=messages,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating HPI prose: {e}"

def ai_double_check_gaps(extracted_data, cleaned_transcript):
    system_prompt = (
        "You are a data validation agent for clinical topic extraction. Compare the extracted topic list against the "
        "original transcript.\n\n"
        "TASKS:\n"
        "1. Identify any topic, detail, stated negative finding, or numeric value present in the transcript that is "
        "missing from the extracted list, and add it as a new entry or append it to an existing matching topic.\n"
        "2. Do not remove or alter any existing entries.\n"
        "3. Do not invent information that is not present in the transcript.\n"
        "4. Keep the same JSON array format: a list of objects with 'topic' and 'details' fields.\n\n"
        "Output raw JSON array directly. No markdown formatting."
    )
    user_payload = {
        "current_topics": extracted_data,
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