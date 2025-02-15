import pandas as pd
import json
import re

# Medical entity detection improvements
MEDICAL_TERMS = {
    "stomach cancer": ["stomach cancer", "gastric cancer"],
    "carcinoma": ["carcinoma", "squamous cell"],
    "diverticulum": ["diverticulum", "pouch"],
    "ovarian cancer": ["ovarian cancer", "surface epithelial-stromal tumor"],
    "pancreatic tumor": ["pancreatic tumor", "pancreas neoplasm"],
    "salivary gland tumor": ["salivary gland tumor"]
}

def extract_medical_term(text):
    text_lower = text.lower()
    # Check for multi-word terms first
    for term, keywords in MEDICAL_TERMS.items():
        if any(f" {kw} " in f" {text_lower} " for kw in keywords):
            return term
    # Then check single-word matches
    medical_words = ["cancer", "tumor", "carcinoma", "diverticulum"]
    for word in medical_words:
        if f" {word} " in f" {text_lower} ":
            return word
    return None

def convert_csv_to_json(csv_path, json_path):
    # Read CSV with medical Q&A pairs
    data = pd.read_csv(csv_path)
    
    # Extract structured medical knowledge
    processed_data = []
    pattern = re.compile(r"\[INST\](.*?)\[/INST\](.*?)</s>", re.DOTALL)
    
    for idx, row in data.iterrows():
        text = row['text']
        try:
            # Extract Q&A using regex
            match = pattern.search(text)
            if not match:
                print(f"Row {idx+1}: Missing INST tags, skipping")
                continue
                
            question = match.group(1).replace("Answer this question truthfully:", "").strip()
            answer = match.group(2).strip()
            
            # Extract medical entity for tag
            question_clean = re.sub(r"\b(what|how|why|when|where|which|who|could|would|should|please|provide|describe|explain)\b", "", question, flags=re.IGNORECASE).strip()
            
            # Extract medical entity using improved detection
            medical_tag = extract_medical_term(question_clean) or extract_medical_term(answer)
            
            if not medical_tag:
                medical_tag = "Medical Condition"  # Default fallback
            
            tag = medical_tag
            
            # Extract symptoms/patterns from answer
            patterns = []
            answer_lower = answer.lower()
            
            # Previous symptom extraction logic
            if "symptoms include" in answer_lower:
                symptom_text = answer.split("symptoms include")[1].split(".")[0]
                patterns = [
                    s.strip().capitalize() 
                    for s in re.split(r", | and ", symptom_text)
                    if s.strip()
                ]
            
            # Fallback to tag if no patterns found
            if not patterns:
                patterns = [tag]
            
            processed_data.append({
                "tag": tag,
                "patterns": patterns,
                "responses": [answer]
            })
            
        except Exception as e:
            print(f"Row {idx+1}: Error processing - {str(e)}")
            print(f"Problematic text: {text[:100]}...")

    # Create intents structure
    intents = {"intents": processed_data}
    
    # Save to JSON
    with open(json_path, 'w') as json_file:
        json.dump(intents, json_file, indent=2)
    
    print(f"Successfully converted {len(processed_data)}/{len(data)} medical entries")

if __name__ == "__main__":
    convert_csv_to_json(
        csv_path="HealthCareFacts/HealthCareFacts.csv",
        json_path="intents.json"
    ) 