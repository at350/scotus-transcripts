import json

filename = 'scotus_corpus.jsonl'
cases = set()
utterances = 0

try:
    with open(filename, 'r') as f:
        for line in f:
            utterances += 1
            try:
                data = json.loads(line)
                cases.add(data.get('docket_number'))
            except:
                pass

    print(f"Total Utterances: {utterances}")
    print(f"Unique Cases: {len(cases)}")
except FileNotFoundError:
    print("File not found.")


# OUTPUT:
# Total Utterances: 201881
# Unique Cases: 1684