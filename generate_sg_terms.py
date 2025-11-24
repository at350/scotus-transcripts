import json
import os
import openai
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit(1)

client = openai.OpenAI(api_key=API_KEY)

# List from preprocess_transcripts.py
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
    "Malcolm L. Stewart",
    "Edwin S. Kneedler",
    "Michael R. Dreeben",
    "Curtis E. Gannon",
    "Eric J. Feigin",
    "Sarah E. Harrington",
    "Brian H. Fletcher",
    "Masha G. Hansford",
    "Frederick Liu",
    "Vivek Suri",
    "Jonathan C. Bond",
    "Sopan Joshi",
    "Austin L. Raynor",
    "Mathew D. Kuhn",
    "Benjamin W. Snyder",
    "Yaira Dubin",
    "Erica L. Ross",
    "Colleen E. Roh Sinzdak",
    "Reedy C. Swanson",
    "Caroline A. Flynn",
    "David M. Morrell",
    "Morgan L. Ratner",
    "Jonathan Y. Ellis",
    "Rachel P. Kovner",
    "Zachary D. Tripp",
    "Hashim M. Mooppan",
    "Christopher G. Michel",
    "Robert A. Parker",
    "Nicole A. Saharsky",
    "Roman Martinez",
    "Allon Kedem",
    "Ginger D. Anders",
    "John F. Bash",
    "Ann O'Connell",
    "Elaine J. Goldenberg",
    "Anthony A. Yang",
    "Melissa Arbus Sherry",
    "Pratik A. Shah",
    "Leondra R. Kruger",
    "Joseph R. Palmore",
    "William M. Jay",
    "Benjamin J. Horwich",
    "Sri Srinivasan",
    "Deanne E. Maynard",
    "Kannon K. Shanmugam",
    "Daryl Joseffer",
    "Lisa S. Blatt",
    "Patricia A. Millett",
    "James A. Feldman",
    "Matthew D. Roberts",
    "Jeffrey P. Minear",
    "Austin C. Schlick",
    "David B. Salmons",
    "Douglas Hallward-Driemeier",
    "Irving L. Gornstein",
    "Jeffrey A. Lamken",
    "Lawrence G. Wallace",
    "Kent L. Jones",
    "Edward C. DuMont",
    "Beth S. Brinkmann",
    "Lisa Schiavo Blatt",
    "Paul R.Q. Wolfson",
    "David C. Frederick",
    "Richard A. Seamon",
    "Cornelia T.L. Pillard",
}

def get_service_years(name):
    prompt = f"""
    Provide the years of service for {name} in the Office of the Solicitor General (OSG) of the United States.
    Include any role (Solicitor General, Deputy, Assistant, etc.).
    If they had multiple stints, include all of them.
    
    Return JSON only:
    {{
        "name": "{name}",
        "terms": [
            {{"start_year": YYYY, "end_year": YYYY}},
            ...
        ],
        "notes": "Brief details on roles"
    }}
    If end year is present (still serving), use 2025.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Using a strong model for knowledge retrieval
            messages=[
                {"role": "system", "content": "You are a legal historian specializing in the US Solicitor General's office."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error for {name}: {e}")
        return None

def main():
    results = {}
    # Deduplicate names (e.g. Seth Waxman vs Seth P. Waxman)
    # We'll query for the longest version if multiple exist, or just query all and merge?
    # Let's just query all unique strings in the set to be safe, then we can map them back.
    
    sorted_names = sorted(list(SG_NAMES))
    
    print(f"Querying terms for {len(sorted_names)} names...")
    
    for name in tqdm(sorted_names):
        info = get_service_years(name)
        if info:
            results[name] = info['terms']
    
    with open("sg_terms.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Saved to sg_terms.json")

if __name__ == "__main__":
    main()
