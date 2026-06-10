import spacy
from spacy.matcher import PhraseMatcher
import re

nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab)

terms = [
    "fever", "chills", "fatigue", "weakness", "cough", "shortness of breath", 
    "dyspnea", "chest pain", "palpitations", "wheezing", "nausea", "vomiting", 
    "diarrhea", "constipation", "abdominal pain", "headache", "dizziness"
]

patterns = [nlp.make_doc(text) for text in terms]
matcher.add("TerminologyList", patterns)

uncountable_symptoms = {
    "shortness of breath", "dyspnea", "fatigue", "weakness", 
    "chills", "palpitations", "wheezing", "nausea", 
    "vomiting", "diarrhea", "constipation", "abdominal pain", "chest pain"
}

severities = ["mild", "moderate", "severe", "sharp", "dull", "crushing", "burning"]
timings = ["constant", "continuous", "intermittent", "episodic", "worsening", "resolving", "fluctuating"]

def extract_symptom_context(sent_text, symptom_name):
    text = sent_text.lower()
    profile = {
        "severity": None,
        "timing": None,
        "duration": None,
        "trigger": None,
        "aggravating": None,
        "alleviating": None,
        "radiation": None
    }
    
    for sev in severities + ["pretty severe", "really bad"]:
        if sev in text:
            profile["severity"] = "severe" if "severe" in sev or "bad" in sev else sev
            break
            
    for tim in timings:
        if tim in text:
            profile["timing"] = tim
            break
    if "comes and goes" in text or "on and off" in text:
        profile["timing"] = "intermittent"

    duration_match = re.search(r'(?:for|since|lasted|started|like)\s+([\w\s]+?(?:days|hours|weeks|months|years|night|morning|evening|ago))', text)
    if duration_match:
        raw_dur = duration_match.group(0).strip()
        if "ago" in raw_dur:
            raw_dur = raw_dur.replace("like ", "").replace("started ", "")
        profile["duration"] = raw_dur
    elif "last night" in text:
        profile["duration"] = "since last night"
        
    rad_match = re.search(r'(?:radiates\s+to|moves\s+to|travels\s+to|goes\s+to|in\s+the)\s+([\w\s]+?(?:arm|leg|back|shoulder|chest|neck|jaw))', text)
    if rad_match:
        matched_phrase = rad_match.group(0).strip()
        if "moves to" in matched_phrase:
            matched_phrase = matched_phrase.replace("moves to", "radiates to")
        profile["radiation"] = matched_phrase
        
    trig_match = re.search(r'(?:triggered\s+by|brought\s+on\s+by|started\s+while|caused\s+by)\s+([\w\s]+?(?:eating|exertion|running|walking|resting|stress|meals))', text)
    if trig_match:
        profile["trigger"] = trig_match.group(0).strip()

    return profile

def generate_hpi(transcribed_text):
    doc = nlp(transcribed_text)
    active_symptoms = []
    negated_symptoms = []
    symptom_profiles = {}
    last_active_symptom = None
    
    for sent in doc.sents:
        sent_text_lower = sent.text.lower()
        sent_doc = nlp(sent.text)
        matches = matcher(sent_doc)
        has_match = False
        
        for match_id, start, end in matches:
            has_match = True
            matched_span = sent_doc[start:end]
            matched_token = matched_span.root
            isNegated = False
            
            for child in matched_token.children:
                if child.dep_ == "neg" or child.text.lower() in ["no", "n't", "not"]:
                    if matched_token.lemma_ not in ["pain", "nausea", "vomiting"]:
                        isNegated = True
                        break
            
            if matched_token.dep_ == "conj":
                for child in matched_token.head.children:
                    if child.dep_ == "neg" or child.text.lower() in ["no", "n't", "not"]:
                        isNegated = True
                        break
            
            for ancestor in matched_token.ancestors:
                if ancestor.lemma_ in ["deny", "negative", "without"]:
                    isNegated = True
                    break
                if ancestor.pos_ == "VERB":
                    for child in ancestor.children:
                        if child.dep_ == "neg" or child.text.lower() in ["n't", "not"]:
                            if child.i < matched_token.i:
                                isNegated = True
                                break
            
            symptom_name = matched_span.text.lower()
            
            if isNegated:
                if symptom_name not in negated_symptoms:
                    negated_symptoms.append(symptom_name)
            else:
                if symptom_name not in active_symptoms:
                    active_symptoms.append(symptom_name)
                last_active_symptom = symptom_name
                
                if symptom_name not in symptom_profiles:
                    symptom_profiles[symptom_name] = extract_symptom_context(sent_text_lower, symptom_name)
                    
        if not has_match and last_active_symptom and (sent_text_lower.strip().startswith("it ") or "it's" in sent_text_lower):
            extra_profile = extract_symptom_context(sent_text_lower, last_active_symptom)
            for key, val in extra_profile.items():
                if val and not symptom_profiles[last_active_symptom][key]:
                    symptom_profiles[last_active_symptom][key] = val

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
            details = []
            
            if profile.get("duration"):
                details.append(profile["duration"])
            if profile.get("trigger"):
                details.append(profile["trigger"])
            if profile.get("radiation"):
                details.append(profile["radiation"])
            if profile.get("aggravating"):
                details.append(profile["aggravating"])
            if profile.get("alleviating"):
                details.append(profile["alleviating"])
                
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