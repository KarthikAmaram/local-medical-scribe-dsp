import spacy
from spacy.matcher import PhraseMatcher


nlp = spacy.load("en_core_web_sm")

matcher = PhraseMatcher(nlp.vocab)
terms = ["cough", "fever", "chest pain", "headache"]

patterns = [nlp.make_doc(text) for text in terms]
matcher.add("TerminologyList", patterns)

doc = nlp("The patient has a slight cough and mild fever.")

matches = matcher(doc)
for match_id, start, end in matches:
    span = doc[start:end]
    print(span.text)