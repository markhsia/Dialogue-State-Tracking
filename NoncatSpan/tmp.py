import json


with open("train.jsonl", 'r') as f1, open("train_.jsonl", 'r') as f2:
    for line1, line2 in zip(f1.readlines(), f2.readlines()):
        d1 = json.loads(line1)
        d2 = json.loads(line2)
        del d1["start"]
        del d1["end"]
        del d2["start"]
        del d2["end"]
        if d1 != d2:
            print(d1)
            print(d2)
            print('\n')
