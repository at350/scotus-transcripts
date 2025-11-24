import json
import os
import re
import nltk
from tqdm import tqdm

# Initialize NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading nltk punkt tokenizer...")
    nltk.download('punkt')
    nltk.download('punkt_tab')


# Configuration
INPUT_DIR = "oyez/cases"
OUTPUT_FILE = "scotus_corpus.jsonl"
START_YEAR = 2000
END_YEAR = 2024

# Load SG Terms
SG_TERMS_FILE = "sg_terms.json"
SG_TERMS = {}
if os.path.exists(SG_TERMS_FILE):
    with open(SG_TERMS_FILE, 'r') as f:
        SG_TERMS = json.load(f)
else:
    print(f"Warning: {SG_TERMS_FILE} not found. SG identification will be limited.")

def normalize_name(name):
    """
    Normalizes a name by removing periods and middle initials if present.
    Example: "Michael R. Dreeben" -> "Michael Dreeben"
    """
    if not name:
        return ""
    # Remove dots
    name = name.replace(".", "")
    parts = name.split()
    if len(parts) > 2:
        # Assume First Middle Last -> First Last
        return f"{parts[0]} {parts[-1]}"
    return name

def is_justice(speaker):
    if not speaker:
        return False
    roles = speaker.get("roles")
    if roles:
        for role in roles:
            if isinstance(role, dict) and role.get("type") == "scotus_justice":
                return True
    return False

def is_active_osg(speaker_name, year):
    """
    Checks if the speaker was active in the OSG during the given year.
    Returns True if:
    1. Name matches a known OSG attorney.
    2. Year falls within one of their service terms.
    """
    if not speaker_name or not year:
        return False
    
    # Try exact match
    terms = SG_TERMS.get(speaker_name)
    
    # Try normalized match
    if not terms:
        norm_name = normalize_name(speaker_name)
        # Search through keys
        for key in SG_TERMS:
            if normalize_name(key) == norm_name:
                terms = SG_TERMS[key]
                break
    
    if not terms:
        return False
        
    # Check years
    for term in terms:
        start = term.get("start_year")
        end = term.get("end_year")
        if start and end:
            if start <= year <= end:
                return True
        elif start: # Open ended or current
             if start <= year:
                 return True
                 
    return False

def clean_text(text):
    # Remove [Laughter], [inaudible], (Applause)
    text = re.sub(r'\[Laughter\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[Inaudible\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(Applause\)', '', text, flags=re.IGNORECASE)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_case_metadata(transcript_filename):
    # Determine case summary filename
    # Pattern: {year}.{docket}-t{xx}.json -> {year}.{docket}.json
    # Or if no -t{xx}, assume it matches or check for base name
    base_name = re.sub(r'-t\d+\.json$', '.json', transcript_filename)
    case_filepath = os.path.join(INPUT_DIR, base_name)
    
    if not os.path.exists(case_filepath):
        return {}

    with open(case_filepath, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def parse_date(date_str):
    # Example: "Oral Argument - November 04, 2020"
    match = re.search(r'(\w+ \d{1,2}, \d{4})', date_str)
    if match:
        return match.group(1)
    return None

def get_speaker_metadata(speaker, advocates, case_summary, year):
    speaker_name = speaker.get("name", "Unknown")
    speaker_id = speaker.get("ID")

    # Default Role/Side/Affiliation
    role = "Private Counsel" # Default
    side = "Unknown"
    affiliation = "Unknown"
    
    # Find advocate in list
    advocate_entry = None
    if advocates:
        for entry in advocates:
            adv = entry.get("advocate")
            if not adv:
                continue
            if str(adv.get("ID")) == str(speaker_id) or adv.get("name") == speaker_name:
                advocate_entry = entry
                break
    
    if advocate_entry:
        desc = advocate_entry.get("advocate_description")
        if desc:
            desc = desc.lower()
        else:
            desc = ""
        
        # Determine Side
        if "petitioner" in desc or "appellant" in desc:
            side = "Petitioner"
        elif "respondent" in desc or "appellee" in desc:
            side = "Respondent"
        elif "amicus" in desc:
            side = "Amicus"
        
        # Determine Affiliation (simple extraction)
        # "for the ..."
        match = re.search(r'for (the )?(.+?)(,|$)', desc)
        if match:
            affiliation = match.group(2).strip()
            # Clean up common prefixes/suffixes
            affiliation = re.sub(r'^(united states|usa|u\.s\.)$', 'United States', affiliation, flags=re.IGNORECASE)
        
        # Determine Role
        # Check if "United States" or "Solicitor General" is in the FULL description
        is_us_affiliation = "united states" in desc.lower() or "solicitor general" in desc.lower()
        
        # Check if speaker is ACTIVE in OSG during this year
        is_active = is_active_osg(speaker_name, year)
        
        if is_us_affiliation:
             if is_active or "general" in speaker_name.lower():
                 role = "Solicitor General" 
             else:
                 role = "Government Counsel"
        elif "state of" in affiliation.lower():
            role = "State Counsel"
        
        # Override for SGs/Deputies/Assistants who are ACTIVE
        # This handles cases where metadata is poor (e.g. "for Respondent") but they are definitely OSG
        if is_active:
            role = "Solicitor General"
            
    return role, side, affiliation

def process_files():
    with open(OUTPUT_FILE, 'w') as outfile:
        files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
        
        for filename in tqdm(files, desc="Processing transcripts"):
            # Filter by year 2000-2024
            try:
                year_str = filename.split('.')[0]
                if '_' in year_str: 
                    continue 
                year = int(year_str)
            except ValueError:
                continue

            if not (START_YEAR <= year <= END_YEAR):
                continue

            filepath = os.path.join(INPUT_DIR, filename)
            with open(filepath, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    continue

            if "transcript" not in data or data["transcript"] is None:
                continue

            transcript = data["transcript"]
            case_name = transcript.get("title", "Unknown")
            
            # Metadata Extraction
            case_metadata = get_case_metadata(filename)
            docket_number = case_metadata.get("docket_number", "Unknown")
            advocates_list = case_metadata.get("advocates", [])
            
            # Date extraction
            transcript_title = data.get("title", "") # Top level title often has date
            argument_date = parse_date(transcript_title)
            
            # Iterate through sections and turns
            sections = transcript.get("sections", [])
            speaking_turn_index = 0
            
            for section in sections:
                turns = section.get("turns", [])
                for turn in turns:
                    speaker = turn.get("speaker")
                    if not speaker:
                        continue
                    
                    speaker_name = speaker.get("name", "Unknown")
                    speaker_id = speaker.get("ID")
                    
                    # Determine Speaker Type (Basic)
                    if is_justice(speaker):
                        continue # Exclude Justices
                    
                    # Enhanced Metadata
                    role, side, affiliation = get_speaker_metadata(speaker, advocates_list, case_metadata, year)
                    
                    # Refine Speaker Type based on Role
                    if role == "Solicitor General":
                        speaker_type = "SG"
                    else:
                        speaker_type = "Other"

                    # Extract and clean text
                    text_blocks = turn.get("text_blocks", [])
                    full_text = " ".join([tb.get("text", "") for tb in text_blocks])
                    cleaned_text = clean_text(full_text)
                    
                    if not cleaned_text:
                        continue

                    # Sentence Segmentation
                    sentences = nltk.sent_tokenize(cleaned_text)

                    # Create Record
                    record = {
                        "case_name": case_name,
                        "year": year,
                        "docket_number": docket_number,
                        "argument_date": argument_date,
                        "speaker_name": speaker_name,
                        "speaker_type": speaker_type,
                        "role": role,
                        "case_side": side,
                        "speaker_affiliation": affiliation,
                        "speaking_turn_index": speaking_turn_index,
                        "utterance_text": cleaned_text.lower(),
                        "sentence_list": sentences
                    }
                    
                    outfile.write(json.dumps(record) + "\n")
                    speaking_turn_index += 1

if __name__ == "__main__":
    process_files()
