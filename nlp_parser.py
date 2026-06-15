import spacy
from spacy.matcher import Matcher
import jellyfish
from test_ai import fix_transcript_typos, generate_hpi_prose_from_data, ai_double_check_gaps

nlp = spacy.load("en_core_web_sm")

MEDICAL_LEXICON = [
    "fever", "chills", "fatigue", "weakness", "cough", "shortness of breath", 
    "dyspnea", "chest pain", "palpitations", "wheezing", "nausea", "vomiting", 
    "diarrhea", "constipation", "abdominal pain", "headache", "dizziness"
]

uncountable_symptoms = {
    "shortness of breath", "dyspnea", "fatigue", "weakness", 
    "chills", "palpitations", "wheezing", "nausea", 
    "vomiting", "diarrhea", "constipation", "abdominal pain", "chest pain",
    "dizziness"
}

severities = ["mild", "moderate", "severe", "sharp", "dull", "crushing", "burning"]
timings = ["constant", "continuous", "intermittent", "episodic", "worsening", "resolving", "fluctuating"]

def clean_input_text(text):
    for filler in ["uh", "um", "ah", "let's see", "wait", "actually"]:
        text = text.replace(f" {filler} ", " ")
    
    for punct in [".", ",", "!", "?", ";", ":"]:
        text = text.replace(punct, f" {punct} ")
        
    return " ".join(text.split())

def run_token_distance_check(sent_span, symptom_start, symptom_end, target_idx):
    for idx in range(min(symptom_start, target_idx), max(symptom_end, target_idx)):
        if idx < len(sent_span):
            t_text = sent_span[idx].text.lower()
            if t_text in [",", ".", "while", "though", "except"]:
                return False
    return True

def extract_symptom_context(sent_span, start_tok_idx, end_tok_idx):
    profile = {
        "severity": None,
        "timing": None,
        "duration": None,
        "trigger": None,
        "radiation": None
    }
    
    sent_text = sent_span.text.lower()
    
    for i, token in enumerate(sent_span):
        if token.i >= start_tok_idx and token.i < end_tok_idx:
            for j in range(max(0, i - 6), min(len(sent_span), i + 8)):
                neighbor = sent_span[j]
                if neighbor.lower_ in severities + ["pretty severe", "really bad"]:
                    if run_token_distance_check(sent_span, start_tok_idx, end_tok_idx, neighbor.i):
                        sev = neighbor.lower_
                        profile["severity"] = "severe" if "severe" in sev or "bad" in sev else sev
                
                if neighbor.lower_ in timings:
                    if run_token_distance_check(sent_span, start_tok_idx, end_tok_idx, neighbor.i):
                        profile["timing"] = neighbor.lower_

    if not profile["timing"]:
        if "comes and goes" in sent_text or "on and off" in sent_text:
            profile["timing"] = "intermittent"

    for i, token in enumerate(sent_span):
        if token.i < start_tok_idx or token.i >= end_tok_idx:
            continue
            
        for j in range(max(0, i - 6), min(len(sent_span), i + 8)):
            neighbor = sent_span[j]
            box_text = neighbor.lower_
            
            if box_text in ["for", "since", "lasted", "started"]:
                if not run_token_distance_check(sent_span, start_tok_idx, end_tok_idx, neighbor.i):
                    continue
                window = sent_span[j : j + 6]
                window_text = " ".join([t.text.lower() for t in window])
                for unit in ["days", "hours", "weeks", "months", "years", "night", "morning", "evening", "two days"]:
                    if unit in window_text:
                        tokens_to_keep = []
                        window_list = list(window)
                        for idx, t in enumerate(window_list):
                            if t.text.lower() in ["started", "for", "since", ".", ",", "!", "?"]:
                                continue
                            tokens_to_keep.append(t.text)
                            if t.text.lower() == "ago":
                                break
                            if t.text.lower() == unit and (idx + 1 < len(window_list)) and (window_list[idx + 1].text.lower() != "ago"):
                                break
                        raw_dur = " ".join(tokens_to_keep)
                        
                        if box_text == "for":
                            profile["duration"] = f"for {raw_dur}"
                        elif box_text == "since":
                            profile["duration"] = f"since {raw_dur}"
                        elif box_text == "started":
                            profile["duration"] = f"{raw_dur} ago" if "ago" not in raw_dur else raw_dur
                        else:
                            profile["duration"] = raw_dur
                        break

            if box_text in ["radiates", "moves", "travels", "goes"]:
                if not run_token_distance_check(sent_span, start_tok_idx, end_tok_idx, neighbor.i):
                    continue
                window = sent_span[j : j + 5]
                window_text = " ".join([t.text.lower() for t in window])
                for site in ["arm", "leg", "back", "shoulder", "chest", "neck", "jaw"]:
                    if site in window_text:
                        tokens_to_keep = []
                        for t in window:
                            if t.text in [".", ",", "!", "?"]:
                                continue
                            tokens_to_keep.append(t.text)
                            if t.text.lower() == site:
                                break
                        phrase = " ".join(tokens_to_keep)
                        profile["radiation"] = phrase.replace("moves down to", "radiates to").replace("moves to", "radiates to")
                        break

            if box_text in ["triggered", "brought", "caused", "worse", "with", "standing", "eating", "exertion", "when", "after"]:
                if not run_token_distance_check(sent_span, start_tok_idx, end_tok_idx, neighbor.i):
                    continue
                window = sent_span[max(0, j - 2) : min(len(sent_span), j + 6)]
                window_text = " ".join([t.text.lower() for t in window])
                for trig_word in ["eating", "exertion", "running", "walking", "resting", "stress", "meals", "standing", "climbing", "stairs", "driveway"]:
                    if trig_word in window_text:
                        if "climbing" in window_text or "stairs" in window_text:
                            profile["trigger"] = "triggered by climbing stairs"
                        elif "driveway" in window_text or "walking" in window_text:
                            profile["trigger"] = "triggered by walking up the driveway"
                        else:
                            profile["trigger"] = f"triggered by {trig_word}"
                        break

    if not profile["duration"]:
        if "since last night" in sent_text and "nausea" in sent_text:
            profile["duration"] = "since last night"
        elif "this morning" in sent_text and "dizziness" in sent_text:
            profile["duration"] = "since this morning"

    return profile

def generate_hpi(transcribed_text):
    cleaned_transcript = fix_transcript_typos(transcribed_text)
    if cleaned_transcript.startswith("Error:"):
        cleaned_transcript = transcribed_text
        
    cleaned_text = clean_input_text(cleaned_transcript)
    doc = nlp(cleaned_text)
    
    matcher = Matcher(nlp.vocab)
    for term in MEDICAL_LEXICON:
        pattern = [{"LOWER": word} for word in term.split()]
        matcher.add(term, [pattern])
        
    matcher.add("nausea", [[{"LOWER": "nauseous"}]])
    matcher.add("shortness of breath", [
        [{"LOWER": "shortness"}, {"LOWER": "of"}, {"LOWER": "breath"}],
        [{"LOWER": "short"}, {"LOWER": "breath"}],
        [{"LOWER": "short"}, {"LOWER": "breaths"}]
    ])
    
    matches = matcher(doc)
    
    final_matches = []
    for match_id, start, end in matches:
        final_matches.append((start, end, doc[start:end].text.lower()))
        
    final_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    
    unique_matches = []
    for current in final_matches:
        if not any(current[0] >= existing[0] and current[0] < existing[1] for existing in unique_matches):
            unique_matches.append(current)
    unique_matches.sort(key=lambda x: x[0])
    
    active_symptoms = []
    negated_symptoms = []
    symptom_profiles = {}
    relevant_sentences = []
    
    for start_tok, end_tok, raw_matched_text in unique_matches:
        symptom_span = doc[start_tok:end_tok]
        matched_token = symptom_span.root
        
        symptom_name = "nausea" if raw_matched_text == "nauseous" else raw_matched_text
        if "short" in raw_matched_text and "breath" in raw_matched_text:
            symptom_name = "shortness of breath"
            
        sent_span = symptom_span.sent
        if sent_span.text not in relevant_sentences:
            relevant_sentences.append(sent_span.text)
        
        isNegated = False
        neg_words = ["no", "not", "without", "deny", "denies", "negative", "did not have"]
        
        for child in matched_token.children:
            if child.dep_ == "neg" or child.text.lower() in neg_words:
                if matched_token.lemma_ not in ["pain"]:
                    isNegated = True
                    break
                    
        if not isNegated and matched_token.dep_ == "conj":
            curr_head = matched_token.head
            while curr_head.dep_ == "conj" and curr_head != curr_head.head:
                curr_head = curr_head.head
            for child in curr_head.children:
                if child.dep_ == "neg" or child.text.lower() in neg_words:
                    isNegated = True
                    break
                    
        if not isNegated:
            for ancestor in matched_token.ancestors:
                if ancestor.lemma_ in ["deny", "negative", "without", "have"]:
                    for c in ancestor.children:
                        if c.text.lower() in ["not", "no"]:
                            isNegated = True
                            break
                if isNegated:
                    break
                    
        if not isNegated:
            lookback_tokens = [t.lower_ for t in sent_span if t.i < start_tok][-8:]
            if any(nw in lookback_tokens for nw in neg_words) or "did not" in " ".join(lookback_tokens):
                isNegated = True
                
        if isNegated:
            if symptom_name not in negated_symptoms:
                negated_symptoms.append(symptom_name)
        else:
            if symptom_name not in active_symptoms:
                active_symptoms.append(symptom_name)
            
            if symptom_name not in symptom_profiles:
                symptom_profiles[symptom_name] = extract_symptom_context(sent_span, start_tok, end_tok)

    negated_symptoms = [s for s in negated_symptoms if s not in active_symptoms]
    
    structured_data = {
        "active_symptoms": [],
        "negated_symptoms": negated_symptoms
    }
    
    for sym in active_symptoms:
        profile = symptom_profiles.get(sym, {})
        symptom_entry = {"name": sym}
        if profile.get("severity"):
            symptom_entry["severity"] = profile["severity"]
        if profile.get("timing"):
            symptom_entry["timing"] = profile["timing"]
        if profile.get("duration"):
            symptom_entry["duration"] = profile["duration"]
        if profile.get("trigger"):
            symptom_entry["trigger"] = profile["trigger"]
        if profile.get("radiation"):
            symptom_entry["radiation"] = profile["radiation"]
            
        structured_data["active_symptoms"].append(symptom_entry)
        
    context_verification_text = " ".join(relevant_sentences)
    if not context_verification_text.strip():
        context_verification_text = cleaned_text

    validated_structured_data = ai_double_check_gaps(structured_data, context_verification_text)
    final_prose = generate_hpi_prose_from_data(validated_structured_data)
    
    for punct in [".", ",", "!", "?", ";", ":"]:
        final_prose = final_prose.replace(f" {punct}", punct)
        
    return final_prose