#from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

import streamlit as st
import spacy

#import spacy

#nlp = spacy.load("en_core_web_sm")

#def extract_entities(text):
    #doc = nlp(text)
    #entities = [(ent.text, ent.label_) for ent in doc.ents]
    #return entities

from transformers import TFAutoModelForTokenClassification, AutoTokenizer, pipeline

model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"

# Load TensorFlow-based model
model = TFAutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
def extract_entities(text):
    return ner_pipeline(text)


st.title('RealTime Named Entity Recognition')

user_input = st.text_area('Enter text for entity recognition')

if st.button("Extract Entities"):
    if user_input.strip():  # Check if input is not empty
        entities = extract_entities(user_input)

        st.write("Raw Output:", entities)  # Debugging Step

        if entities:
            st.subheader("Extracted Entities:")
            for entity in entities:
                entity_word = entity.get("word", "N/A")  # Default to 'N/A' if missing
                entity_label = entity.get("entity_group", entity.get("entity", "N/A"))  # Try both keys
                entity_score = entity.get("score", 0)

                st.write(f"**{entity_word}** -> `{entity_label}` (Score: {entity_score:.2f})")
        else:
            st.warning("No named entities found. Try a longer sentence.")
    else:
        st.warning("Please enter some text for entity recognition.")






