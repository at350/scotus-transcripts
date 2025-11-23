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

# Curated list of Solicitors General and Acting SGs (2000-2024)
# Matches against full names in Oyez data
SG_NAMES = {
    "Seth P. Waxman", "Seth Waxman",
    "Barbara D. Underwood", "Barbara Underwood",
    "Theodore B. Olson", "Theodore Olson", "Ted Olson",
    "Paul D. Clement", "Paul Clement",
    "Gregory G. Garre", "Gregory Garre",
    "Elena Kagan",
    "Neal K. Katyal", "Neal Katyal",
    "Donald B. Verrilli, Jr.", "Donald Verrilli", "Donald B. Verrilli",
    "Ian H. Gershengorn", "Ian Gershengorn",
    "Noel J. Francisco", "Noel Francisco",
    "Jeffrey B. Wall", "Jeffrey Wall",
    "Elizabeth B. Prelogar", "Elizabeth Prelogar",
    "Scott G. Stewart", # Deputy SG often arguing? No, he was MS SG.
    "Malcolm L. Stewart", # Deputy SG, very frequent.
    "Edwin S. Kneedler", # Deputy SG, very frequent.
    "Michael R. Dreeben", # Deputy SG, very frequent.
    "Curtis E. Gannon", # Deputy SG
    "Eric J. Feigin", # Deputy SG
    "Sarah E. Harrington", # Deputy SG
    "Brian H. Fletcher", # Deputy SG
    "Masha G. Hansford", # Assistant to SG
    "Frederick Liu", # Assistant to SG
    "Vivek Suri", # Assistant to SG
    "Jonathan C. Bond", # Assistant to SG
    "Sopan Joshi", # Assistant to SG
    "Austin L. Raynor", # Assistant to SG
    "Mathew D. Kuhn", # Assistant to SG
    "Benjamin W. Snyder", # Assistant to SG
    "Yaira Dubin", # Assistant to SG
    "Erica L. Ross", # Assistant to SG
    "Colleen E. Roh Sinzdak", # Assistant to SG
    "Reedy C. Swanson", # Assistant to SG
    "Caroline A. Flynn", # Assistant to SG
    "David M. Morrell", # Assistant to SG
    "Morgan L. Ratner", # Assistant to SG
    "Jonathan Y. Ellis", # Assistant to SG
    "Rachel P. Kovner", # Assistant to SG
    "Zachary D. Tripp", # Assistant to SG
    "Hashim M. Mooppan", # Counselor to SG
    "Christopher G. Michel", # Assistant to SG
    "Robert A. Parker", # Assistant to SG
    "Nicole A. Saharsky", # Assistant to SG
    "Roman Martinez", # Assistant to SG
    "Allon Kedem", # Assistant to SG
    "Sarah E. Harrington", # Assistant to SG
    "Ginger D. Anders", # Assistant to SG
    "Rachel P. Kovner", # Assistant to SG
    "John F. Bash", # Assistant to SG
    "Ann O'Connell", # Assistant to SG
    "Elaine J. Goldenberg", # Assistant to SG
    "Sarah E. Harrington", # Assistant to SG
    "Curtis E. Gannon", # Assistant to SG
    "Eric J. Feigin", # Assistant to SG
    "Anthony A. Yang", # Assistant to SG
    "Melissa Arbus Sherry", # Assistant to SG
    "Pratik A. Shah", # Assistant to SG
    "Leondra R. Kruger", # Assistant to SG
    "Joseph R. Palmore", # Assistant to SG
    "William M. Jay", # Assistant to SG
    "Benjamin J. Horwich", # Assistant to SG
    "Sri Srinivasan", # Principal Deputy SG
    "Deanne E. Maynard", # Assistant to SG
    "Kannon K. Shanmugam", # Assistant to SG
    "Daryl Joseffer", # Principal Deputy SG
    "Lisa S. Blatt", # Assistant to SG
    "Patricia A. Millett", # Assistant to SG
    "James A. Feldman", # Assistant to SG
    "Matthew D. Roberts", # Assistant to SG
    "Jeffrey P. Minear", # Assistant to SG
    "Austin C. Schlick", # Assistant to SG
    "David B. Salmons", # Assistant to SG
    "Douglas Hallward-Driemeier", # Assistant to SG
    "Irving L. Gornstein", # Assistant to SG
    "Jeffrey A. Lamken", # Assistant to SG
    "Matthew D. Roberts", # Assistant to SG
    "James A. Feldman", # Assistant to SG
    "Malcolm L. Stewart", # Deputy SG
    "Edwin S. Kneedler", # Deputy SG
    "Michael R. Dreeben", # Deputy SG
    "Lawrence G. Wallace", # Deputy SG
    "Kent L. Jones", # Assistant to SG
    "Edward C. DuMont", # Assistant to SG
    "Beth S. Brinkmann", # Assistant to SG
    "Lisa Schiavo Blatt", # Assistant to SG
    "Jeffrey A. Lamken", # Assistant to SG
    "Paul R.Q. Wolfson", # Assistant to SG
    "David C. Frederick", # Assistant to SG
    "Richard A. Seamon", # Assistant to SG
    "Cornelia T.L. Pillard", # Assistant to SG
}

def is_justice(speaker):
    if not speaker:
        return False
    roles = speaker.get("roles")
    if roles:
        for role in roles:
            if isinstance(role, dict) and role.get("type") == "scotus_justice":
                return True
    return False

def is_sg(speaker_name):
    if not speaker_name:
        return False
    # Check exact match or if name contains "General" (though Oyez name field usually doesn't have title)
    # Oyez name: "Lori H. Windham"
    return speaker_name in SG_NAMES

def clean_text(text):
    # Remove [Laughter], [inaudible], (Applause)
    text = re.sub(r'\[Laughter\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[Inaudible\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(Applause\)', '', text, flags=re.IGNORECASE)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_files():
    with open(OUTPUT_FILE, 'w') as outfile:
        files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
        
        for filename in tqdm(files, desc="Processing transcripts"):
            # Filter by year 2000-2024
            # Filename format: {year}.{docket}.json or {year}.{docket}-t{xx}.json
            try:
                year_str = filename.split('.')[0]
                if '_' in year_str: # Handle 1900_1940 style if present, though usually just year
                    continue # These are likely older
                year = int(year_str)
            except ValueError:
                continue

            if not (START_YEAR <= year <= END_YEAR):
                continue

            # Only process transcript files (usually have -tXX or just .json if it contains transcript?)
            # README says: "Transcripts... appending t01...".
            # But some files like `2020.19-123.json` (no t01) might be case summaries?
            # Let's check if it has "transcript" field.
            
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
            
            # Iterate through sections and turns
            sections = transcript.get("sections", [])
            for section in sections:
                turns = section.get("turns", [])
                for turn in turns:
                    speaker = turn.get("speaker")
                    if not speaker:
                        continue
                    
                    speaker_name = speaker.get("name", "Unknown")
                    
                    # Determine Speaker Type
                    if is_justice(speaker):
                        continue # Exclude Justices
                    elif is_sg(speaker_name):
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
                        "speaker_name": speaker_name,
                        "speaker_type": speaker_type,
                        "utterance_text": cleaned_text.lower(), # Normalization: lowercase
                        "sentence_list": sentences
                    }
                    
                    outfile.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    process_files()
