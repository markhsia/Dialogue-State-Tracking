import glob
import json


with open("cat_results.json", 'r') as f:
    data = json.load(f)
    print(len(data))
