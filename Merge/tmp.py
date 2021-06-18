import glob
import json


dial_ids = set()
with open("results.csv", 'r') as f:
    for line in f:
        dial_id, _ = line.split(',')
        dial_ids.add(dial_id)


for fn in glob.glob("../datasets/data/test_unseen/*.json"):
    with open(fn, 'r') as f:
        data = json.load(f)
    
    for d in data:
        if d["dialogue_id"] not in dial_ids:
            print(fn, d["dialogue_id"])


