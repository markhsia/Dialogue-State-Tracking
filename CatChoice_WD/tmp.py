import json
import numpy as np

X = []
with open("train.jsonl", 'r') as f:
    for line in f:
        d = json.loads(line)
        X.append(d["wide_features"] + [d["label"]])
X = np.array(X).T
print(np.corrcoef(X)[-1])
