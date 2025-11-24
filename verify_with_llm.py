import json
import os
import random
import openai
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
INPUT_FILE = "scotus_corpus.jsonl"
SAMPLE_SIZE = 10
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit(1)

client = openai.OpenAI(api_key=API_KEY)

def get_llm_verification(case_name, year, speaker_name, text):
    prompt = f"""
    You are a legal expert on the US Supreme Court.
    
    Case: {case_name} ({year})
    Speaker: {speaker_name}
    Excerpt: "{text[:200]}..."
    
    Question: Was {speaker_name} representing the United States (e.g., as Solicitor General, Deputy SG, Assistant to SG) in oral argumentation in this specific case?
    
    Answer with JSON only:
    {{
        "is_representing_us": true/false,
        "reasoning": "brief explanation"
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a helpful legal assistant."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error calling OpenAI: {e}")
        return None

def verify_data():
    print("Loading corpus...")
    data = []
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Filter for relevant roles to sample
    sg_speakers = [d for d in data if d['speaker_type'] == 'SG']
    other_speakers = [d for d in data if d['speaker_type'] == 'Other']
    
    # Sample balanced set
    sample_sg = random.sample(sg_speakers, min(len(sg_speakers), SAMPLE_SIZE // 2))
    sample_other = random.sample(other_speakers, min(len(other_speakers), SAMPLE_SIZE // 2))
    
    sample = sample_sg + sample_other
    random.shuffle(sample)
    
    print(f"Verifying {len(sample)} records...")
    
    correct_count = 0
    discrepancies = []
    
    for record in tqdm(sample):
        llm_result = get_llm_verification(
            record['case_name'],
            record['year'],
            record['speaker_name'],
            record['utterance_text']
        )
        
        if not llm_result:
            continue
            
        is_sg_dataset = record['speaker_type'] == 'SG'
        is_sg_llm = llm_result['is_representing_us']
        
        if is_sg_dataset == is_sg_llm:
            correct_count += 1
        else:
            discrepancies.append({
                "record": record,
                "llm_reasoning": llm_result['reasoning'],
                "dataset_type": record['speaker_type'],
                "llm_says_us": is_sg_llm
            })
            
    accuracy = (correct_count / len(sample)) * 100
    print(f"\nVerification Complete.")
    print(f"Agreement Rate: {accuracy:.2f}%")
    
    if discrepancies:
        print("\nDiscrepancies found:")
        for d in discrepancies:
            print(f"- Case: {d['record']['case_name']} ({d['record']['year']})")
            print(f"  Speaker: {d['record']['speaker_name']}")
            print(f"  Dataset: {d['dataset_type']}")
            print(f"  LLM says representing US: {d['llm_says_us']}")
            print(f"  Reasoning: {d['llm_reasoning']}")
            print("-" * 40)

if __name__ == "__main__":
    verify_data()
