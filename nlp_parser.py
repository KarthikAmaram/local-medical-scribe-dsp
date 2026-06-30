import spacy
import re
import time
import difflib
from test_ai import fix_transcript_typos, extract_topics, generate_hpi_prose_from_data, ai_double_check_gaps

DEBUG = False

import sys
import os

def _load_spacy_model():
    import spacy
    if getattr(sys, 'frozen', False):
        model_path = os.path.join(sys._MEIPASS, 'en_core_web_sm')
        return spacy.load(model_path)
    return spacy.load('en_core_web_sm')

nlp = _load_spacy_model()

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

DRUG_SUFFIX_EXCLUSIONS = {
    "cholesterol", "alcohol", "menthol", "protocol", "control", "patrol",
    "symbol", "scandal", "cathedral", "hospital", "vital", "capital",
}

KNOWN_DRUG_NAMES = BRAND_NAMES | GENERIC_NAMES
MULTIWORD_MATCH_THRESHOLD = 0.72
MULTIWORD_STOPWORDS = {
    "a", "an", "the", "his", "her", "their", "its", "and", "or", "but",
    "on", "off", "for", "from", "with", "without", "to", "of", "in",
    "is", "was", "are", "were", "be", "been", "being",
    "taking", "takes", "took", "take", "started", "starting", "start",
    "stopped", "stopping", "stop", "continued", "continuing", "continues",
    "continue", "since", "after", "before", "denies", "reports", "still",
    "also", "currently", "previously", "now", "daily", "nightly", "weekly",
}

def _best_drug_name_match(candidate):
    best_ratio = 0.0
    best_name = None
    for name in KNOWN_DRUG_NAMES:
        ratio = difflib.SequenceMatcher(None, candidate, name).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_name = name
    return best_ratio, best_name

def _detect_drug_names(text, original_transcript):
    found = []
    seen = set()
    orig_lower = original_transcript.lower()

    for match in re.finditer(r'\b[A-Za-z][a-zA-Z0-9-]{3,}\b', text):
        word = match.group(0)
        word_lower = word.lower()
        if word_lower in seen:
            continue

        is_known_name = word_lower in BRAND_NAMES or word_lower in GENERIC_NAMES
        is_suffix_match = (
            word_lower not in DRUG_SUFFIX_EXCLUSIONS
            and any(word_lower.endswith(suffix) for suffix in DRUG_SUFFIXES)
        )

        if not is_known_name and not is_suffix_match:
            continue

        flagged_phrase = word
        flag_start = match.start()

        if is_suffix_match and not is_known_name:
            preceding_text = text[:match.start()]
            preceding_match = re.search(r'([A-Za-z][a-zA-Z0-9-]{2,}),?\s+$', preceding_text)
            if preceding_match:
                preceding_word = preceding_match.group(1)
                if preceding_word.lower() not in MULTIWORD_STOPWORDS:
                    combined = (preceding_word + word_lower).lower()
                    ratio, best_name = _best_drug_name_match(combined)
                    if ratio >= MULTIWORD_MATCH_THRESHOLD and preceding_word.lower() not in BRAND_NAMES and preceding_word.lower() not in GENERIC_NAMES:
                        flagged_phrase = preceding_word + " " + word
                        flag_start = preceding_match.start(1)

        phrase_lower = flagged_phrase.lower()
        if phrase_lower in seen:
            continue
        seen.add(phrase_lower)
        seen.add(word_lower)

        if flagged_phrase != word:
            found.append({
                "word": flagged_phrase,
                "severity": "hallucinated",
                "reason": f'"{flagged_phrase}" looks like a garbled or split medication name — verify against the chart.'
            })
            continue

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

def _remove_redundant_drug_suffix_echo(text):
    for name in KNOWN_DRUG_NAMES:
        for suffix in DRUG_SUFFIXES:
            if name.endswith(suffix) and len(suffix) < len(name):
                pattern = r'\b(' + re.escape(name) + r')\s+' + re.escape(suffix) + r'\b'
                text = re.sub(pattern, r'\1', text, flags=re.IGNORECASE)
    return text

WHISPER_PHONETIC_CORRECTIONS = [
    (r'\brecheck\s+lives\b', 'recheck labs'),
    (r'\bcheck\s+lives\b', 'check labs'),
    (r'\bup\s+today\b(?=\s+with)', 'up to date'),
    (r'\blevo\s+dioroxide\b', 'levothyroxine'),
    (r'\blevo\s+thyroxide\b', 'levothyroxine'),
    (r'\btorvus,?\s+statin\b', 'atorvastatin'),
    (r'\btorvis,?\s+statin\b', 'atorvastatin'),
]

def clean_input_text(text):
    for pattern, replacement in WHISPER_PHONETIC_CORRECTIONS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
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

    word_numbers = {
        r'\bone\b': '1', r'\btwo\b': '2', r'\bthree\b': '3',
        r'\bfour\b': '4', r'\bfive\b': '5', r'\bsix\b': '6',
        r'\bseven\b': '7', r'\beight\b': '8', r'\bnine\b': '9',
        r'\bten\b': '10', r'\beleven\b': '11', r'\btwelve\b': '12',
    }
    for pattern, digit in word_numbers.items():
        text = re.sub(pattern, digit, text, flags=re.IGNORECASE)

    return text

_VAGUE_TAIL_EXCLUDE = (
    r'(?!'
    r'\d'
    r'|(?:glucose|sodium|potassium|chloride|bicarbonate|creatinine|bun|hemoglobin|hematocrit'
    r'|hgb|hct|wbc|rbc|plt|tsh|t3|t4|a1c|ldl|hdl|triglycerides|cholesterol'
    r'|pressure|pulse|rate|temp|temperature|spo2|o2|sat|bmi|weight|height'
    r'|mg|dl|mmol|mcg|meq|iu|units?|kg|lbs?|cm|mm|bpm)\b'
    r')'
)

VAGUE_PATTERNS = [
    r'a\s+(?:little|bit|tad)\s+' + _VAGUE_TAIL_EXCLUDE + r'\w+',
    r'kind\s+of\s+' + _VAGUE_TAIL_EXCLUDE + r'\w+',
    r'sort\s+of\s+' + _VAGUE_TAIL_EXCLUDE + r'\w+',
    r'somewhat\s+' + _VAGUE_TAIL_EXCLUDE + r'\w+',
    r'slightly\s+' + _VAGUE_TAIL_EXCLUDE + r'\w+',
    r'(?:I\s+)?think\s+(?:it\s+(?:is|was|might\s+be)|maybe|perhaps)\b[^.]*',
    r'maybe\s+' + _VAGUE_TAIL_EXCLUDE + r'\w+',
    r'not\s+sure\s+(?:about|if|whether)[^.]*',
    r'might\s+be\s+' + _VAGUE_TAIL_EXCLUDE + r'\w+',
    r'possibly\s+' + _VAGUE_TAIL_EXCLUDE + r'\w+',
    r'(?:running|been|been\s+running)\s+(?:a\s+little|kind\s+of|sort\s+of|somewhat|slightly)\s+' + _VAGUE_TAIL_EXCLUDE + r'\w+',
]
VAGUE_REGEX = re.compile('|'.join(VAGUE_PATTERNS), re.IGNORECASE)

def _locate_flagged_spans(final_prose, validated_topics):
    spans = []
    prose_lower = final_prose.lower()

    for topic in validated_topics:
        if not topic.get("uncertain"):
            continue
        details = topic.get("details", "")
        if not details:
            continue

        vague_matches = list(VAGUE_REGEX.finditer(details))

        if vague_matches:
            for vm in vague_matches:
                phrase = vm.group(0).strip()
                phrase_lower = phrase.lower()
                idx = prose_lower.find(phrase_lower)
                if idx != -1:
                    spans.append({
                        "start": idx,
                        "end": idx + len(phrase),
                        "topic": topic.get("topic", ""),
                        "reason": topic.get("reason") or "Wording here is ambiguous - double check."
                    })
        else:
            matcher = difflib.SequenceMatcher(None, prose_lower, details.lower(), autojunk=False)
            match = matcher.find_longest_match(0, len(prose_lower), 0, len(details))
            coverage = match.size / max(len(details), 1)
            matched_text = prose_lower[match.a:match.a + match.size].strip()
            if match.size >= MIN_FLAG_MATCH_CHARS and coverage >= 0.6:
                spans.append({
                    "start": match.a,
                    "end": match.a + match.size,
                    "topic": topic.get("topic", ""),
                    "reason": topic.get("reason") or "Wording here is ambiguous - double check."
                })

    spans.sort(key=lambda s: s["start"])
    return spans

def _match_case(replacement, original):
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement

def _enforce_pronoun_gender(text, patient_gender):
    if patient_gender not in ("female", "male"):
        return text

    if patient_gender == "female":
        text = re.sub(r'\bhimself\b', lambda m: _match_case("herself", m.group(0)), text, flags=re.IGNORECASE)
        text = re.sub(r'\bhim\b', lambda m: _match_case("her", m.group(0)), text, flags=re.IGNORECASE)
        text = re.sub(r'\bhis\b', lambda m: _match_case("her", m.group(0)), text, flags=re.IGNORECASE)
        text = re.sub(r'\bhe\b', lambda m: _match_case("she", m.group(0)), text, flags=re.IGNORECASE)
    else:
        text = re.sub(r'\bherself\b', lambda m: _match_case("himself", m.group(0)), text, flags=re.IGNORECASE)
        text = re.sub(r'\bhers\b', lambda m: _match_case("his", m.group(0)), text, flags=re.IGNORECASE)
        text = re.sub(r'\bher\b(?=\s+\w)', lambda m: _match_case("his", m.group(0)), text, flags=re.IGNORECASE)
        text = re.sub(r'\bher\b(?!\s+\w)', lambda m: _match_case("him", m.group(0)), text, flags=re.IGNORECASE)
        text = re.sub(r'\bshe\b', lambda m: _match_case("he", m.group(0)), text, flags=re.IGNORECASE)

    return text

def _detect_patient_gender(text):
    text_lower = text.lower()
    female_count = len(re.findall(r'\b(she|her|hers)\b', text_lower))
    male_count = len(re.findall(r'\b(he|him|his)\b', text_lower))
    if female_count > male_count:
        return "female"
    if male_count > female_count:
        return "male"
    return "unspecified"

def _format_cleaned_transcript(text):
    text = re.sub(r'\s+', ' ', text).strip()
    if text:
        text = text[0].upper() + text[1:]
    text = re.sub(
        r'([.!?])\s+([a-z])',
        lambda m: m.group(1) + ' ' + m.group(2).upper(),
        text
    )
    return text

def generate_hpi(transcribed_text):
    if not transcribed_text or len(transcribed_text.strip()) < 5:
        return transcribed_text, "No clinical dictation captured to parse.", [], []

    pipeline_start = time.time()

    t0 = time.time()
    cleaned_transcript = fix_transcript_typos(transcribed_text)
    if DEBUG:
        print(f"[TIMING] fix_transcript_typos: {time.time() - t0:.2f}s")
    if cleaned_transcript.startswith("Error:"):
        cleaned_transcript = transcribed_text

    cleaned_transcript = _format_cleaned_transcript(cleaned_transcript)

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
    shorthand_text = _remove_redundant_drug_suffix_echo(shorthand_text)

    validated_topics = []
    patient_gender = "unspecified"
    try:
        t0 = time.time()
        topics = extract_topics(shorthand_text)
        if DEBUG:
            print(f"[TIMING] extract_topics: {time.time() - t0:.2f}s")

        if not topics:
            final_prose = shorthand_text
        else:
            t0 = time.time()
            validated_topics = ai_double_check_gaps(topics, shorthand_text)
            if DEBUG:
                print(f"[TIMING] ai_double_check_gaps: {time.time() - t0:.2f}s")

            if not validated_topics:
                validated_topics = topics

            patient_gender = _detect_patient_gender(shorthand_text)

            t0 = time.time()
            final_prose = generate_hpi_prose_from_data(validated_topics, patient_gender)
            if DEBUG:
                print(f"[TIMING] generate_hpi_prose_from_data: {time.time() - t0:.2f}s")
    except Exception:
        final_prose = shorthand_text

    final_prose = _enforce_pronoun_gender(final_prose, patient_gender)

    for punct in [".", ",", "!", "?", ";", ":"]:
        final_prose = final_prose.replace(f" {punct}", punct)

    flagged_spans = _locate_flagged_spans(final_prose, validated_topics) if validated_topics else []
    drug_flags = _detect_drug_names(final_prose, cleaned_transcript)

    if DEBUG:
        print(f"[TIMING] generate_hpi total: {time.time() - pipeline_start:.2f}s")

    return cleaned_transcript, final_prose, flagged_spans, drug_flags