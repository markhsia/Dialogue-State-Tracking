import os
import json
import argparse
import glob
from collections import defaultdict
import random

random.seed(1114)

def get_state_updates(prev_state, curr_state):
    state_updates = dict(curr_state)
    for slot, values in curr_state.items():
        if slot in prev_state and sorted(prev_state[slot]) == sorted(values):
            del state_updates[slot]

    return state_updates

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dial_dirs", nargs='+', required=True, type=str)
    parser.add_argument("-s", "--schema_file", required=True, type=str)
    parser.add_argument("-o", "--out_file", required=True, type=str)
    parser.add_argument("-l", "--with_labels", action="store_true")
    args = parser.parse_args()

    with open(args.schema_file, 'r') as rf:
        schema = json.load(rf)
    cat_descriptions = defaultdict(dict)
    cat_slots = defaultdict(set)
    cat_poss_values = defaultdict(dict)
    for service_i, service_chunk in enumerate(schema):
        service = service_chunk["service_name"]
        cat_descriptions[service]["service_desc"] = service_chunk["description"]
        cat_descriptions[service]["slot_descs"] = dict()
        for slot_i, slot_chunk in enumerate(service_chunk["slots"]):
            if not slot_chunk["is_categorical"]:
                continue
            slot = slot_chunk["name"]
            cat_slots[service].add(slot)
            cat_descriptions[service]["slot_descs"][slot] = slot_chunk["description"]
            poss_values = ["unknown", "dontcare"] + slot_chunk["possible_values"]
            cat_poss_values[service][slot] = poss_values

    data = []
    total_slots_count = 0
    for dial_dir in args.dial_dirs:
        for fn in sorted(glob.glob(os.path.join(dial_dir, "dialogues_*.json"))):
            with open(fn, 'r') as rf:
                dials = json.load(rf)
            
            for dial in dials:
                dial_id = dial["dialogue_id"]
                services = dial["services"]
                utterances = ''
                states_record = {service: dict() for service in services}
                bounds_record = {service: defaultdict(dict) for service in services}
                for turn in dial["turns"]:
                    utterances += turn["speaker"].strip().capitalize() + ": "
                    utterance = turn["utterance"]
                    if args.with_labels:
                        for frame in turn["frames"]:
                            service = frame["service"]
                            if service not in states_record:
                                continue
                            if "state" in frame:
                                curr_state = frame["state"]["slot_values"]
                                state_updates = get_state_updates(states_record[service], curr_state)
                                for slot, values in state_updates.items():
                                    for value in values:
                                        bounds_record[service][slot][value] = (len(utterances), \
                                                                        len(utterances) + len(utterance))
                                states_record[service] = curr_state

                    utterances += utterance + ' '

                for service in services:
                    service_desc = cat_descriptions[service]["service_desc"]
                    states = states_record[service]
                    for slot, poss_values in sorted(cat_poss_values[service].items()):
                        slot_desc = cat_descriptions[service]["slot_descs"][slot]
                        if args.with_labels:
                            true_value = states.get(slot, ["unknown"])[0]
                        for poss_value in poss_values:
                            if args.with_labels:
                                if poss_value == true_value:
                                    label = 1
                                    start, end = bounds_record[service][slot].get(true_value, (-1, -1))
                                else:
                                    label = 0
                                    start, end = -1, -1
                                if (start, end) == (-1, -1) and label == 1 and poss_value != "unknown":
                                    print("Not matched: {} | {} | {} | {} | {}".format(fn, dial_id, \
                                                                                    service, slot, poss_value))
                                    continue
                            else:
                                label = -1
                                start, end = -1, -1
                            data.append({"id": total_slots_count,
                                        "dial_id": dial_id,
                                        "utterances": utterances,
                                        "service": service,
                                        "service_desc": service_desc,
                                        "slot": slot,
                                        "slot_desc": slot_desc,
                                        "value": poss_value,
                                        "label": label,
                                        "start": start,
                                        "end": end})
                        total_slots_count += 1
                    
    pos_count = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    with open(args.out_file, 'w') as wf:
        for d in data:
            print(json.dumps(d), file=wf)
            pos_count += d["label"]
    print(pos_count / len(data))
