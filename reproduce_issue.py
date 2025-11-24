import sys
import os

# Add current directory to path to import preprocess_transcripts
sys.path.append(os.getcwd())

from preprocess_transcripts import get_speaker_metadata

def test_lisa_blatt_classification():
    print("Testing Lisa S. Blatt classification...")
    
    # Mock data based on the user's report
    speaker = {
        "name": "Lisa S. Blatt",
        "ID": "12345" # Dummy ID
    }
    
    # Mock advocates list as it might appear in Oyez data
    # We need to simulate what Oyez would provide for Lisa Blatt in this case.
    # Usually it would contain her name and a description like "for the Respondent"
    advocates = [
        {
            "advocate": {
                "name": "Lisa S. Blatt",
                "ID": "12345"
            },
            "advocate_description": "for the Respondent"
        }
    ]
    
    case_summary = {} # Not used for role determination in the current function logic
    
    role, side, affiliation = get_speaker_metadata(speaker, advocates, case_summary)
    
    print(f"Role: {role}")
    print(f"Side: {side}")
    print(f"Affiliation: {affiliation}")
    
    if role == "Solicitor General":
        print("FAIL: Lisa S. Blatt misclassified as Solicitor General")
    else:
        print("SUCCESS: Lisa S. Blatt correctly classified as non-SG")

if __name__ == "__main__":
    test_lisa_blatt_classification()
