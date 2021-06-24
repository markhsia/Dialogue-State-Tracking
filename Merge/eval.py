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
    parser.add_argument("-s", "--schema_file", required=True, type=str)
    parser.add_argument("-i", "--in_file", required=True, type=str)
    args = parser.parse_args()
    
    with open(args.schema_file, 'r') as rf:
        schema = json.load(rf)
    noncat_slots = defaultdict(set)
    for service_i, service_chunk in enumerate(schema):
        service = service_chunk["service_name"]
        for slot_i, slot_chunk in enumerate(service_chunk["slots"]):
            if slot_chunk["is_categorical"]:
                continue
            slot = slot_chunk["name"]
            noncat_slots[service].add(slot)

    gt = dict()
    dial_count = 0
    for fn in sorted(glob.glob(os.path.join(args.dial_dir, "dialogues_*.json"))):
        with open(fn, 'r') as rf:
            dials = json.load(rf)
            
        for dial in dials:
            dial_count += 1
            dial_id = dial["dialogue_id"]
            utterances = ' '.join([turn["utterance"] for turn in dial["turns"]]).lower()
            states_record = dict()
            for turn in dial["turns"]:
                for frame in turn["frames"]:
                    service = frame["service"]
                    if "state" in frame:
                        states_record[service] = frame["state"]["slot_values"]
            gt_dict = dict()
            for service, slot_values in states_record.items():
                for slot, values in slot_values.items():
                    values = [v.lower() for v in values]
                    value = values[0]
                    if slot in noncat_slots[service]:
                        for v in values:
                            if v in utterances or v == "dontcare":
                                value = v
                                break
                        if value == "dontcare":
                            print(fn, dial_id, service, slot)
                    value = value.replace(',', '_')
                    gt_dict["{}-{}".format(service, slot).lower()] = value
            gt_list = []
            for slot, value in sorted(gt_dict.items(), key=lambda x: x[0]):
                gt_list.append("{}={}".format(slot, value))
            gt[dial_id] = '|'.join(gt_list)
    
    joint_correct_sum = 0
    joint_total_sum = 0
    with open(args.in_file, 'r') as f:
        lines = f.readlines()[1:]
    assert len(lines) == dial_count, dial_count
    for line in lines:
        joint_total_sum += 1
        dial_id, line = line.strip().split(',')
        if line == gt.get(dial_id, "None"):
            joint_correct_sum += 1
    
    print("JGA = {:.5f}".format(joint_correct_sum / joint_total_sum))
