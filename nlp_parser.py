import spacy
from spacy.matcher import PhraseMatcher


nlp = spacy.load("en_core_web_sm")

matcher = PhraseMatcher(nlp.vocab)
terms = ["cough", "fever", "chest pain", "headache"]

patterns = [nlp.make_doc(text) for text in terms]
matcher.add("TerminologyList", patterns)

doc = nlp("The patient has a headache but no fever or cough.")

matches = matcher(doc)

for match_id, start, end in matches:
    matched_token = doc[start]
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
        print(f"Symptom: {matched_token.text} (NEGATED)")
    else:
        print(f"Symptom: {matched_token.text} (ACTIVE)")