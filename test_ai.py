import json
from openai import OpenAI  # type: ignore

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

KEEP_ALIVE = "30m"

def warm_up_model(model="llama3"):
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            temperature=0.0,
            extra_body={"keep_alive": KEEP_ALIVE}
        )
        return True
    except Exception:
        return False

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
    }
]

PREAMBLE_PHRASES = [
    "Here is the corrected text:",
    "Here's the corrected text:",
    "Corrected text:",
    "Here is the corrected transcript:",
    "Here's the corrected transcript:",
    "Corrected transcript:",
    "Here is the text:",
    "Here's the text:",
]

def _strip_preamble(text):
    for preamble in PREAMBLE_PHRASES:
        if text.lower().startswith(preamble.lower()):
            return text[len(preamble):].strip()
    return text

def _fix_transcript_typos_full_rewrite(raw_transcript):
    system_prompt = (
        "You are a strict clinical spelling and typo correction utility. Your only job is to fix literal spelling mistakes, "
        "run-on words, and microphone phonetic errors (e.g., convert 'coff' to 'cough').\n\n"
        "UNIVERSAL CORRECTION PROTOCOLS:\n"
        "1. PHONETIC CONTEXT MATCHING: If a word is mangled or unrecognized, analyze its syllable sounds alongside surrounding clinical indicators (e.g., 'weekly', 'injection', 'mg', 'blood sugar'). You MUST map the unknown word to a medication that matches BOTH the phonetic sound AND the clinical context. Never swap a phonetically distant word just because it is a more common drug (e.g., do not replace a word sounding like 'mocharo' with 'metformin' if the text says 'injection').\n"
        "2. PRESERVE UNKNOWN MEDICAL TERMS: If an unrecognized word cannot be reliably matched phonetically to a known clinical term, leave the misspelled word EXACTLY as it was transcribed. Do NOT guess generic English words (like converting 'Moujaro' to 'Mystery') or substitute different medical conditions.\n"
        "2b. KNOWN PHRASE-LEVEL PHONETIC CONFUSIONS: certain common clinical phrases are frequently mis-transcribed as a different but similarly-sounding phrase, most often 'up to date' being heard as 'up today'. If a sentence containing 'up today' is discussing screenings, vaccines, immunizations, or labs, correct it to 'up to date'. Only apply this correction when context clearly supports it; do not alter 'today' when used in its normal sense (e.g., 'patient is here today').\n"
        "3. CONVERT WORD NUMBERS: Convert any spelled-out numbers into standard digits.\n"
        "4. DO NOT REPHRASE: Do not reorder, rephrase, summarize, or remove any clinical content. This rule applies to every single word. Do NOT substitute one grammatically valid word for another even if the result sounds more natural — for example, do not change 'since' to 'and', 'after' to 'following', 'but' to 'however', or any similar substitution. Only fix clear spelling errors, not style.\n"
        "5. NEVER SUBSTITUTE DRUG NAMES: This is the most critical rule. If you see any medication name, you must never replace it with a different medication name, even if you think the different name is more common or more familiar. Lisinopril is not interchangeable with LIPITOR. Losartan is not interchangeable with lovastatin. These are completely different drugs. If a medication name looks like it might be a phonetic error, only correct the spelling of that same drug — never substitute a different drug. When in doubt, leave the word exactly as transcribed.\n"
        "6. NEVER INVENT FACTS: Do not add dates, doses, durations, or any clinical detail not present word-for-word in the input. If the input does not mention a date, your output must not mention a date.\n"
        "7. Output ONLY the clean text string. No conversational notes, no preamble like 'Here is the corrected text:', no commentary, no explanation. Your entire response must be the corrected transcript and nothing else."
    )
    try:
        response = client.chat.completions.create(
            model="llama3",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_transcript}
            ],
            temperature=0.0,
            extra_body={"keep_alive": KEEP_ALIVE}
        )
        result = response.choices[0].message.content.strip()
        return _strip_preamble(result)
    except Exception as e:
        return f"Error: {e}"

def _fix_transcript_typos_diff(raw_transcript):
    system_prompt = (
        "You are a strict clinical spelling and typo correction utility. Your only job is to find literal spelling "
        "mistakes, run-on words, and microphone phonetic errors in the transcript (e.g., 'coff' should become "
        "'cough'), and report them as a list of corrections rather than rewriting the whole transcript.\n\n"
        "Respond with a JSON array only, in this exact shape: "
        '[{"wrong": "exact text as it appears in the transcript", "fixed": "corrected text"}]\n'
        "If nothing needs correcting, respond with an empty array: []\n\n"
        "RULES:\n"
        "1. The 'wrong' field must be copied EXACTLY, character for character, from the input transcript — same "
        "capitalization, same spacing, same punctuation. This is critical: it will be used for an exact text search.\n"
        "2. PHONETIC CONTEXT MATCHING: If a word is mangled or unrecognized, analyze its syllable sounds alongside "
        "surrounding clinical indicators (e.g., 'weekly', 'injection', 'mg', 'blood sugar'). You MUST map the unknown "
        "word to a medication that matches BOTH the phonetic sound AND the clinical context. Never swap a "
        "phonetically distant word just because it is a more common drug.\n"
        "3. PRESERVE UNKNOWN MEDICAL TERMS: If an unrecognized word cannot be reliably matched phonetically to a "
        "known clinical term, do NOT include it as a correction. Do NOT guess generic English words or substitute "
        "different medical conditions.\n"
        "4. KNOWN PHRASE-LEVEL PHONETIC CONFUSIONS: certain common clinical phrases are frequently mis-transcribed "
        "as a different but similarly-sounding phrase, most often 'up to date' being heard as 'up today'. If a "
        "sentence containing 'up today' is discussing screenings, vaccines, immunizations, or labs, correct it to "
        "'up to date'. Only apply this correction when context clearly supports it.\n"
        "5. CONVERT WORD NUMBERS: Convert any spelled-out numbers into standard digits.\n"
        "6. DO NOT REPHRASE: Never include a correction that just rewords a phrase to sound more natural — for "
        "example, do not change 'since' to 'and', 'after' to 'following', 'but' to 'however'. Only include actual "
        "spelling/phonetic errors, never style changes.\n"
        "7. NEVER SUBSTITUTE DRUG NAMES: This is the most critical rule. Never propose changing one medication name "
        "to a different medication name, even if you think the different name is more common. Lisinopril is not "
        "interchangeable with LIPITOR. Losartan is not interchangeable with lovastatin. If a medication name looks "
        "like a phonetic error, only correct the spelling of that same drug — never substitute a different drug.\n"
        "8. NEVER INVENT FACTS: Do not propose any correction that adds a date, dose, duration, or any clinical "
        "detail not already present word-for-word in the input.\n"
        "9. Each 'wrong' value must appear in the transcript only once. If the same mistake occurs multiple times, "
        "list each occurrence as a separate correction with enough surrounding context in 'wrong' to make it unique."
    )
    try:
        response = client.chat.completions.create(
            model="llama3",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_transcript}
            ],
            temperature=0.0,
            extra_body={"keep_alive": KEEP_ALIVE}
        )
        clean_content = response.choices[0].message.content.strip()
        if clean_content.startswith("```json"):
            clean_content = clean_content.split("```json")[1].split("```")[0].strip()
        elif clean_content.startswith("```"):
            clean_content = clean_content.split("```")[1].split("```")[0].strip()
        return json.loads(clean_content)
    except Exception:
        return None

def _apply_corrections(original_text, corrections):
    result = original_text
    for correction in corrections:
        wrong = correction.get("wrong", "")
        fixed = correction.get("fixed", "")
        if not wrong:
            return None
        occurrences = result.count(wrong)
        if occurrences != 1:
            return None
        result = result.replace(wrong, fixed, 1)
    return result

def fix_transcript_typos(raw_transcript):
    normalized = " ".join(raw_transcript.split())
    corrections = _fix_transcript_typos_diff(normalized)
    if corrections is not None:
        applied = _apply_corrections(normalized, corrections)
        if applied is not None:
            print(f"[PATH] fix_transcript_typos: diff ({len(corrections)} correction(s))")
            return applied
    print("[PATH] fix_transcript_typos: fallback (full rewrite)")
    return _fix_transcript_typos_full_rewrite(normalized)

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
            temperature=0.0,
            extra_body={"keep_alive": KEEP_ALIVE}
        )
        clean_content = response.choices[0].message.content.strip()
        if clean_content.startswith("```json"):
            clean_content = clean_content.split("```json")[1].split("```")[0].strip()
        elif clean_content.startswith("```"):
            clean_content = clean_content.split("```")[1].split("```")[0].strip()
        return json.loads(clean_content)
    except Exception:
        return [{"topic": "Unparsed transcript", "details": cleaned_transcript}]

def generate_hpi_prose_from_data(extracted_data, patient_gender="unspecified"):
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
        "7. Output ONLY the finalized clinical text.\n\n"
        + _gender_directive(patient_gender)
    )
    messages = [{"role": "system", "content": system_prompt}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": json.dumps(ex["topics"])})
        messages.append({"role": "assistant", "content": ex["text"]})
    prose_input = [{"topic": t.get("topic", ""), "details": t.get("details", "")} for t in extracted_data]
    messages.append({"role": "user", "content": json.dumps(prose_input)})

    try:
        response = client.chat.completions.create(
            model="llama3",
            messages=messages,
            temperature=0.0,
            extra_body={"keep_alive": KEEP_ALIVE}
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating HPI prose: {e}"

def _gender_directive(patient_gender):
    if patient_gender == "female":
        return "PATIENT GENDER FOR THIS NOTE: female. Use she/her/hers for every pronoun referring to the patient. Never use he/him/his, regardless of what pronouns appear in the earlier examples."
    if patient_gender == "male":
        return "PATIENT GENDER FOR THIS NOTE: male. Use he/him/his for every pronoun referring to the patient. Never use she/her/hers, regardless of what pronouns appear in the earlier examples."
    return "PATIENT GENDER FOR THIS NOTE: unspecified. Do not use any gendered pronoun (he/him/his/she/her/hers). Refer to the patient only as 'Pt'."

def ai_double_check_gaps(extracted_data, cleaned_transcript):
    system_prompt = (
        "You are a data validation agent for clinical topic extraction. Compare the extracted topic list against the "
        "original transcript.\n\n"
        "The extracted topics are provided as a numbered list (index starting at 0). Do NOT re-output the existing "
        "topics. Respond with a compact JSON object only, in this exact shape:\n"
        '{"new_topics": [{"topic": "...", "details": "...", "uncertain": true/false, "reason": "..."}], '
        '"flagged_indices": [{"index": 0, "reason": "..."}]}\n\n'
        "TASKS:\n"
        "1. NEW TOPICS: If the transcript contains a topic, detail, stated negative finding, or numeric value not "
        "covered by any existing indexed topic, add it as an entry in 'new_topics'. If nothing is missing, "
        "'new_topics' must be an empty list.\n"
        "2. Do not invent information that is not present in the transcript.\n"
        "3. FLAGGED INDICES: For each EXISTING indexed topic (not new ones), include it in 'flagged_indices' only if "
        "at least one of these specific conditions applies: (a) a value is vague rather than exact, e.g. 'a little "
        "high' or 'some improvement', with no actual number given; (b) the transcript uses hedge language about it "
        "like 'I think', 'maybe', 'not sure', or 'might be'; (c) it conflicts with another statement about the same "
        "topic; (d) it contains a word that looks like a garbled or unresolved transcription error rather than a "
        "real clinical term. Do NOT include an index in 'flagged_indices' if none of these apply — most topics will "
        "not be flagged. Do not flag things just because they are clinically significant; only flag actual ambiguity "
        "in the wording itself.\n"
        "4. For each item in 'flagged_indices', 'reason' must be a short plain-English note (under 12 words) of "
        "exactly what to double check.\n"
        "5. For each item in 'new_topics', set 'uncertain' to true only under the same conditions as rule 3, with a "
        "matching 'reason'; otherwise 'uncertain' is false and 'reason' is an empty string.\n\n"
        "Output ONLY the JSON object described above. No markdown formatting, no commentary."
    )
    indexed_topics = [
        {"index": i, "topic": t.get("topic", ""), "details": t.get("details", "")}
        for i, t in enumerate(extracted_data)
    ]
    user_payload = {
        "indexed_topics": indexed_topics,
        "transcript": cleaned_transcript
    }
    try:
        response = client.chat.completions.create(
            model="llama3",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)}
            ],
            temperature=0.0,
            extra_body={"keep_alive": KEEP_ALIVE}
        )
        clean_content = response.choices[0].message.content.strip()
        if clean_content.startswith("```json"):
            clean_content = clean_content.split("```json")[1].split("```")[0].strip()
        elif clean_content.startswith("```"):
            clean_content = clean_content.split("```")[1].split("```")[0].strip()
        result = json.loads(clean_content)

        merged = [dict(t, uncertain=False, reason="") for t in extracted_data]

        for flag in result.get("flagged_indices", []):
            idx = flag.get("index")
            if isinstance(idx, int) and 0 <= idx < len(merged):
                merged[idx]["uncertain"] = True
                merged[idx]["reason"] = flag.get("reason", "") or "Wording here is ambiguous - double check."

        new_topics = _normalize_flag_fields(result.get("new_topics", []))
        merged.extend(new_topics)

        normalized = _normalize_flag_fields(merged)
        if extracted_data and not normalized:
            return _normalize_flag_fields(extracted_data)
        return normalized
    except Exception:
        return _normalize_flag_fields(extracted_data)

def _normalize_flag_fields(topics):
    normalized = []
    for entry in topics:
        if not isinstance(entry, dict):
            continue
        uncertain = bool(entry.get("uncertain", False))
        normalized.append({
            "topic": entry.get("topic", ""),
            "details": entry.get("details", ""),
            "uncertain": uncertain,
            "reason": entry.get("reason", "") if uncertain else ""
        })
    return normalized