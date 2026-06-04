import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")

matcher = PhraseMatcher(nlp.vocab)
terms = [
    "fever", "chills", "fatigue", "weakness", "cough", "shortness of breath", 
    "dyspnea", "chest pain", "palpitations", "wheezing", "nausea", "vomiting", 
    "diarrhea", "constipation", "abdominal pain", "headache", "dizziness"
]

patterns = [nlp.make_doc(text) for text in terms]
matcher.add("TerminologyList", patterns)

doc = nlp("Patient complains of a cough, and I suspect a mild bronchitis because of the wheezing, though it is reassuring that they do not manifest any fever, chills, or explicit shortness of breath at this moment.")
matches = matcher(doc)

active_symptoms = []
negated_symptoms = []

uncountable_symptoms = {
    "shortness of breath", "dyspnea", "fatigue", "weakness", 
    "chills", "palpitations", "wheezing", "nausea", 
    "vomiting", "diarrhea", "constipation", "abdominal pain", "chest pain"
}

for match_id, start, end in matches:
    matched_span = doc[start:end]
    matched_token = matched_span.root
    isNegated = False

    for child in matched_token.children:
        if child.dep_ == "neg" or child.text.lower() in ["no", "n't", "not"]:
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

    if isNegated:
        negated_symptoms.append(matched_span.text)
    else:
        active_symptoms.append(matched_span.text)

hpi_prose = "The patient presents with "

if active_symptoms:
    if len(active_symptoms) == 1:
        active_str = active_symptoms[0]
    elif len(active_symptoms) == 2:
        active_str = " and ".join(active_symptoms)
    else:
        active_str = ", ".join(active_symptoms[:-1]) + ", and " + active_symptoms[-1]
    
    if active_symptoms[0].lower() in uncountable_symptoms:
        hpi_prose += f"{active_str}. "
    else:
        hpi_prose += f"a {active_str}. "
else:
    hpi_prose += "no acute or major symptoms. "

if negated_symptoms:
    if len(negated_symptoms) == 1:
        negated_str = negated_symptoms[0]
    elif len(negated_symptoms) == 2:
        negated_str = " or ".join(negated_symptoms)
    else:
        negated_str = ", ".join(negated_symptoms[:-1]) + ", or " + negated_symptoms[-1]
        
    hpi_prose += f"They deny any {negated_str}."

print(hpi_prose)