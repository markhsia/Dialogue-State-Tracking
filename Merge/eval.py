import os
import json
import argparse
import glob
from collections import defaultdict
import random

random.seed(1114)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dial_dir", required=True, type=str)
    parser.add_argument("-i", "--in_file", required=True, type=str)
    args = parser.parse_args()
    
    gt = defaultdict(dict)
    for fn in sorted(glob.glob(os.path.join(args.dial_dir, "dialogues_*.json"))):
        with open(fn, 'r') as rf:
            dials = json.load(rf)
            
        for dial in dials:
            dial_id = dial["dialogue_id"]
            for turn in dial["turns"]:
                for frame in turn["frames"]:
                    service = frame["service"]
                    if "state" in frame:
                        curr_state = frame["state"]["slot_values"]
                        for slot, values in curr_state.items():
                            gt[dial_id]["{}-{}".format(service, slot).lower()] = [v.replace(',', '_').lower() for v in values]
    
    joint_correct_sum = 0
    joint_total_sum = 0
    with open(args.in_file, 'r') as f:
        for line in f.readlines()[1:]:
            joint_total_sum += 1
            dial_id, line = line.strip().split(',')
            if dial_id == "16_00025":
                continue
            slices = line.split('|')
            slots, slot_values = [], []
            for s in slices:
                slot, value = s.split('=')
                slots.append(slot)
                slot_values.append((slot, value))
            joint_correct = 1
            if sorted(slots) != sorted(gt[dial_id].keys()):
                joint_correct = 0
                continue
            for s, v in slot_values:
                if v not in gt[dial_id][s]:
                    joint_count = 0
            joint_correct_sum += joint_correct
    
    print("JGA = {:.5f}".format(joint_correct_sum / joint_total_sum))
