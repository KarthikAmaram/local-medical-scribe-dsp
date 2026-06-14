import spacy
from spacy.matcher import Matcher
import jellyfish

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
    return " ".join(text.split())

def extract_symptom_context(sent_span, start_tok_idx, end_tok_idx):
    profile = {
        "severity": None,
        "timing": None,
        "duration": None,
        "trigger": None,
        "aggravating": None,
        "alleviating": None,
        "radiation": None
    }
    
    sent_text = sent_span.text.lower()
    
    for sev in severities + ["pretty severe", "really bad"]:
        if sev in sent_text:
            profile["severity"] = "severe" if "severe" in sev or "bad" in sev else sev
            break
            
    for tim in timings:
        if tim in sent_text:
            profile["timing"] = tim
            break
    if "comes and goes" in sent_text or "on and off" in sent_text:
        profile["timing"] = "intermittent"

    for i, token in enumerate(sent_span):
        if token.i < start_tok_idx or token.i >= end_tok_idx:
            continue
            
        for j in range(max(0, i - 4), min(len(sent_span), i + 6)):
            neighbor = sent_span[j]
            tok_text = neighbor.lower_
            
            if tok_text in ["for", "since", "lasted", "started"]:
                window = sent_span[j : j + 5]
                window_text = " ".join([t.text.lower() for t in window])
                for unit in ["days", "hours", "weeks", "months", "years", "night", "morning", "evening"]:
                    if unit in window_text:
                        tokens_to_keep = []
                        window_list = list(window)
                        for idx, t in enumerate(window_list):
                            if t.text.lower() in ["started", "for", "since"]:
                                continue
                            tokens_to_keep.append(t.text)
                            if t.text.lower() == "ago":
                                break
                            if t.text.lower() == unit and (idx + 1 < len(window_list)) and (window_list[idx + 1].text.lower() != "ago"):
                                break
                        raw_dur = " ".join(tokens_to_keep)
                        
                        if tok_text == "for":
                            profile["duration"] = f"for {raw_dur}"
                        elif tok_text == "since":
                            profile["duration"] = f"since {raw_dur}"
                        elif tok_text == "started":
                            profile["duration"] = f"{raw_dur} ago" if "ago" not in raw_dur else raw_dur
                        else:
                            profile["duration"] = raw_dur
                        break

            if tok_text in ["radiates", "moves", "travels", "goes"]:
                window = sent_span[j : j + 5]
                window_text = " ".join([t.text.lower() for t in window])
                for site in ["arm", "leg", "back", "shoulder", "chest", "neck", "jaw"]:
                    if site in window_text:
                        tokens_to_keep = []
                        for t in window:
                            tokens_to_keep.append(t.text)
                            if t.text.lower() == site:
                                break
                        phrase = " ".join(tokens_to_keep)
                        profile["radiation"] = phrase.replace("moves down to", "radiates to").replace("moves to", "radiates to")
                        break

            if tok_text in ["triggered", "brought", "caused", "worse", "with", "standing", "eating", "exertion"]:
                window = sent_span[max(0, j - 1) : min(len(sent_span), j + 3)]
                window_text = " ".join([t.text.lower() for t in window])
                for trig_word in ["eating", "exertion", "running", "walking", "resting", "stress", "meals", "standing"]:
                    if trig_word in window_text:
                        profile["trigger"] = f"triggered by {trig_word}"
                        break

    if not profile["duration"]:
        if "since last night" in sent_text and "nausea" in sent_text:
            profile["duration"] = "since last night"
        elif "this morning" in sent_text and "dizziness" in sent_text:
            profile["duration"] = "since this morning"

    return profile

def generate_hpi(transcribed_text):
    cleaned_text = clean_input_text(transcribed_text)
    doc = nlp(cleaned_text)
    
    matcher = Matcher(nlp.vocab)
    
    for term in MEDICAL_LEXICON:
        pattern = [{"LOWER": word} for word in term.split()]
        matcher.add(term, [pattern])
        
    matcher.add("nausea", [[{"LOWER": "nauseous"}]])
    matcher.add("shortness of breath", [[{"LOWER": "shortness"}, {"LOWER": "of"}, {"LOWER": "breath"}]])
    
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
    
    for start_tok, end_tok, raw_matched_text in unique_matches:
        symptom_span = doc[start_tok:end_tok]
        matched_token = symptom_span.root
        
        symptom_name = "nausea" if raw_matched_text == "nauseous" else raw_matched_text
        sent_span = symptom_span.sent
        
        isNegated = False
        neg_words = ["no", "not", "without", "deny", "denies", "negative"]
        
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
                if ancestor.lemma_ in ["deny", "negative", "without"]:
                    isNegated = True
                    break
                    
        if not isNegated:
            lookback_tokens = [t.lower_ for t in sent_span if t.i < start_tok][-5:]
            if any(nw in lookback_tokens for nw in neg_words):
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
    sentences = []
    
    if active_symptoms:
        intro_parts = []
        for sym in active_symptoms:
            profile = symptom_profiles.get(sym, {})
            sev = profile.get("severity")
            tim = profile.get("timing")
            
            desc = ""
            if sev: desc += sev
            if tim: desc += f" {tim}" if desc else tim
            
            if sym in uncountable_symptoms:
                intro_parts.append(f"{desc} {sym}" if desc else sym)
            else:
                intro_parts.append(f"a {desc} {sym}" if desc else f"a {sym}")
        
        if len(intro_parts) == 1:
            sentences.append(f"The patient presents with {intro_parts[0]}.")
        elif len(intro_parts) == 2:
            sentences.append(f"The patient presents with {intro_parts[0]} and {intro_parts[1]}.")
        else:
            sentences.append(f"The patient presents with {', '.join(intro_parts[:-1])}, and {intro_parts[-1]}.")
            
        for sym in active_symptoms:
            profile = symptom_profiles.get(sym, {})
            
            duration_str = ""
            if profile.get("duration"):
                dur = profile["duration"]
                if "ago" in dur:
                    duration_str = f"began {dur}"
                elif "since" in dur or "for" in dur:
                    duration_str = f"has been present {dur}"
                else:
                    duration_str = f"has been present for {dur}"
            
            details = []
            if duration_str:
                details.append(duration_str)
            if profile.get("trigger"):
                details.append(profile["trigger"])
            if profile.get("radiation"):
                details.append(profile["radiation"])
                
            if details:
                sym_cap = sym.capitalize()
                if len(details) == 1:
                    sentences.append(f"{sym_cap} {details[0]}.")
                elif len(details) == 2:
                    sentences.append(f"{sym_cap} {details[0]} and {details[1]}.")
                else:
                    sentences.append(f"{sym_cap} {', '.join(details[:-1])}, and {details[-1]}.")
    else:
        sentences.append("The patient presents with no acute or major symptoms.")
        
    if negated_symptoms:
        if len(negated_symptoms) == 1:
            negated_str = negated_symptoms[0]
        elif len(negated_symptoms) == 2:
            negated_str = " or ".join(negated_symptoms)
        else:
            negated_str = ", ".join(negated_symptoms[:-1]) + ", or " + negated_symptoms[-1]
        sentences.append(f"They explicitly deny any {negated_str}.")
        
    return " ".join(sentences)