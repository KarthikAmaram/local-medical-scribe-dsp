import spacy
import re
from test_ai import fix_transcript_typos, extract_topics, generate_hpi_prose_from_data, ai_double_check_gaps

nlp = spacy.load("en_core_web_sm")

def clean_input_text(text):
    for filler in ["uh", "um", "ah", "let's see", "wait", "actually"]:
        text = text.replace(f" {filler} ", " ")
    return " ".join(text.split())

def normalize_to_shorthand(text):
    text = re.sub(r'\b(\d+)\s+over\s+(\d+)\b', r'\1/\2', text, flags=re.IGNORECASE)

    text = re.sub(r'\b(the patient|patient)\b', 'Pt', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(follow[\s-]?up|following[\s-]?up|f\s*/\s*u)\b', 'f/u', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(type 2 diabetes|type two diabetes|diabetes type two)\b', 'DM2', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(up to date)\b', 'UTD', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(complains of|complaining of|c / o)\b', 'c/o', text, flags=re.IGNORECASE)

    text = re.sub(r'\bfour times\b', '4x', text, flags=re.IGNORECASE)
    text = re.sub(r'\bthree times\b', '3x', text, flags=re.IGNORECASE)
    text = re.sub(r'\btwice\b', '2x', text, flags=re.IGNORECASE)
    text = re.sub(r'\bonce\b', '1x', text, flags=re.IGNORECASE)

    return text

def generate_hpi(transcribed_text):
    if not transcribed_text or len(transcribed_text.strip()) < 5:
        return transcribed_text, "No clinical dictation captured to parse."

    cleaned_transcript = fix_transcript_typos(transcribed_text)
    if cleaned_transcript.startswith("Error:"):
        cleaned_transcript = transcribed_text

    cleaned_text = clean_input_text(cleaned_transcript)

    doc = nlp(cleaned_text)
    processed_sentences = []

    for sent in doc.sents:
        sent_str = sent.text.strip()
        if not sent_str:
            continue

        compressed_sent = normalize_to_shorthand(sent_str)
        processed_sentences.append(compressed_sent)

    shorthand_text = " ".join(processed_sentences)

    try:
        topics = extract_topics(shorthand_text)
        validated_topics = ai_double_check_gaps(topics, shorthand_text)
        final_prose = generate_hpi_prose_from_data(validated_topics)
    except Exception:
        final_prose = shorthand_text

    for punct in [".", ",", "!", "?", ";", ":"]:
        final_prose = final_prose.replace(f" {punct}", punct)

    return cleaned_transcript, final_prose