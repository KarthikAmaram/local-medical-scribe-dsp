import spacy
import re
import time
import difflib
from test_ai import fix_transcript_typos, extract_topics, generate_hpi_prose_from_data, ai_double_check_gaps

nlp = spacy.load("en_core_web_sm")

MIN_FLAG_MATCH_CHARS = 10

DRUG_SUFFIXES = (
    "pril", "sartan", "statin", "olol", "dipine", "prazole", "tidine",
    "mycin", "cillin", "cycline", "oxacin", "floxacin", "zepam", "zolam",
    "pam", "lam", "mab", "nib", "tinib", "zumab", "xumab", "umab",
    "gliptin", "gliflozin", "glutide", "parib", "ciclib", "rafenib",
    "setron", "triptan", "lukast", "terol", "phylline", "dronate",
    "oxetine", "pamine", "azine", "ridone", "apine", "tidyl"
)

BRAND_NAMES = {
    "lipitor", "crestor", "zocor", "pravachol", "lescol",
    "norvasc", "zestril", "prinivil", "altace", "accupril",
    "diovan", "cozaar", "avapro", "benicar", "micardis",
    "toprol", "coreg", "lopressor", "tenormin",
    "glucophage", "januvia", "tradjenta", "invokana", "jardiance", "farxiga",
    "ozempic", "victoza", "trulicity", "mounjaro", "wegovy",
    "lantus", "humalog", "novolog", "levemir", "basaglar", "toujeo",
    "synthroid", "levothroid", "armour",
    "nexium", "prilosec", "prevacid", "protonix", "aciphex",
    "zantac", "pepcid", "tagamet",
    "advil", "motrin", "aleve", "tylenol",
    "xanax", "valium", "ativan", "klonopin",
    "zoloft", "prozac", "paxil", "lexapro", "celexa", "effexor", "cymbalta",
    "abilify", "seroquel", "risperdal", "zyprexa",
    "adderall", "ritalin", "concerta", "vyvanse",
    "coumadin", "xarelto", "eliquis", "pradaxa",
    "plavix", "brilinta", "effient",
    "humira", "enbrel", "remicade", "stelara", "dupixent",
    "lyrica", "neurontin", "topamax", "lamictal", "keppra", "depakote",
    "singular", "singulair", "spiriva", "advair", "symbicort", "dulera",
    "singular", "flovent", "pulmicort", "albuterol", "proventil", "ventolin",
}

GENERIC_NAMES = {
    "metformin", "insulin", "aspirin", "acetaminophen", "ibuprofen", "naproxen",
    "warfarin", "clopidogrel", "furosemide", "hydrochlorothiazide", "spironolactone",
    "digoxin", "amiodarone", "diltiazem", "verapamil", "nifedipine",
    "levothyroxine", "prednisone", "hydrocortisone", "methylprednisolone",
    "omeprazole", "pantoprazole", "ranitidine", "famotidine",
    "gabapentin", "pregabalin", "tramadol", "morphine", "oxycodone", "hydrocodone",
    "fentanyl", "codeine", "methadone",
    "lithium", "valproate", "carbamazepine", "phenytoin", "levetiracetam",
    "sertraline", "fluoxetine", "citalopram", "escitalopram", "bupropion",
    "venlafaxine", "duloxetine", "trazodone", "mirtazapine",
    "risperidone", "quetiapine", "olanzapine", "aripiprazole", "haloperidol",
    "methotrexate", "hydroxychloroquine", "sulfasalazine", "azathioprine",
    "allopurinol", "colchicine",
    "tamsulosin", "finasteride", "sildenafil", "tadalafil",
    "azithromycin", "amoxicillin", "doxycycline", "ciprofloxacin",
    "metronidazole", "clindamycin", "vancomycin", "trimethoprim",
    "albuterol", "ipratropium", "montelukast", "fluticasone", "budesonide",
    "tiotropium", "theophylline",
    "lisinopril", "losartan", "valsartan", "atorvastatin", "simvastatin",
    "rosuvastatin", "metoprolol", "atenolol", "carvedilol", "amlodipine",
    "hydralazine", "clonidine", "isosorbide",
}

def _detect_drug_names(text, original_transcript):
    found = []
    words = re.findall(r'\b[A-Za-z][a-zA-Z0-9-]{3,}\b', text)
    seen = set()
    orig_lower = original_transcript.lower()

    for word in words:
        word_lower = word.lower()
        if word_lower in seen:
            continue

        is_drug = (
            any(word_lower.endswith(suffix) for suffix in DRUG_SUFFIXES)
            or word_lower in BRAND_NAMES
            or word_lower in GENERIC_NAMES
        )
        if not is_drug:
            continue

        seen.add(word_lower)

        in_original = bool(re.search(r'\b' + re.escape(word_lower) + r'\b', orig_lower))
        similar_in_original = any(
            word_lower[:max(4, len(word_lower) - 2)] in orig_lower
            for _ in [1]
        )

        if not in_original and not similar_in_original:
            severity = "hallucinated"
            reason = f'"{word}" does not appear in the original transcript — may be a substituted drug name.'
        else:
            severity = "review"
            reason = f'"{word}" is a medication name — verify spelling and dose are correct.'

        found.append({
            "word": word,
            "severity": severity,
            "reason": reason
        })

    return found

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

def _locate_flagged_spans(final_prose, validated_topics):
    spans = []
    prose_lower = final_prose.lower()

    for topic in validated_topics:
        if not topic.get("uncertain"):
            continue
        details = topic.get("details", "")
        if not details:
            continue

        matcher = difflib.SequenceMatcher(None, prose_lower, details.lower(), autojunk=False)
        match = matcher.find_longest_match(0, len(prose_lower), 0, len(details))

        required_len = min(MIN_FLAG_MATCH_CHARS, max(6, int(len(details) * 0.3)))
        if match.size >= required_len:
            spans.append({
                "start": match.a,
                "end": match.a + match.size,
                "topic": topic.get("topic", ""),
                "reason": topic.get("reason") or "Wording here is ambiguous - double check."
            })

    spans.sort(key=lambda s: s["start"])
    return spans

def _detect_patient_gender(text):
    text_lower = text.lower()
    female_count = len(re.findall(r'\b(she|her|hers)\b', text_lower))
    male_count = len(re.findall(r'\b(he|him|his)\b', text_lower))
    if female_count > male_count:
        return "female"
    if male_count > female_count:
        return "male"
    return "unspecified"

def generate_hpi(transcribed_text):
    if not transcribed_text or len(transcribed_text.strip()) < 5:
        return transcribed_text, "No clinical dictation captured to parse.", [], []

    pipeline_start = time.time()

    t0 = time.time()
    cleaned_transcript = fix_transcript_typos(transcribed_text)
    print(f"[TIMING] fix_transcript_typos: {time.time() - t0:.2f}s")
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

    validated_topics = []
    try:
        t0 = time.time()
        topics = extract_topics(shorthand_text)
        print(f"[TIMING] extract_topics: {time.time() - t0:.2f}s")

        if not topics:
            final_prose = shorthand_text
        else:
            t0 = time.time()
            validated_topics = ai_double_check_gaps(topics, shorthand_text)
            print(f"[TIMING] ai_double_check_gaps: {time.time() - t0:.2f}s")

            if not validated_topics:
                validated_topics = topics

            patient_gender = _detect_patient_gender(shorthand_text)

            t0 = time.time()
            final_prose = generate_hpi_prose_from_data(validated_topics, patient_gender)
            print(f"[TIMING] generate_hpi_prose_from_data: {time.time() - t0:.2f}s")
    except Exception:
        final_prose = shorthand_text

    for punct in [".", ",", "!", "?", ";", ":"]:
        final_prose = final_prose.replace(f" {punct}", punct)

    flagged_spans = _locate_flagged_spans(final_prose, validated_topics) if validated_topics else []
    drug_flags = _detect_drug_names(final_prose, transcribed_text)

    print(f"[TIMING] generate_hpi total: {time.time() - pipeline_start:.2f}s")

    return cleaned_transcript, final_prose, flagged_spans, drug_flags