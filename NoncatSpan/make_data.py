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
    parser.add_argument("--keep_no_matched", action="store_true")
    args = parser.parse_args()


    with open(args.schema_file, 'r') as rf:
        schema = json.load(rf)
    noncat_descriptions = defaultdict(dict)
    for service_i, service_chunk in enumerate(schema):
        service = service_chunk["service_name"]
        noncat_descriptions[service]["service_desc"] = service_chunk["description"]
        noncat_descriptions[service]["slot_descs"] = dict()
        for slot_i, slot_chunk in enumerate(service_chunk["slots"]):
            if slot_chunk["is_categorical"]:
                continue
            slot = slot_chunk["name"]
            noncat_descriptions[service]["slot_descs"][slot] = slot_chunk["description"]

    
    data = []
    for dial_dir in args.dial_dirs:
        for fn in sorted(glob.glob(os.path.join(dial_dir, "dialogues_*.json"))):
            with open(fn, 'r') as rf:
                dials = json.load(rf)
            
            for dial in dials:
                dial_id = dial["dialogue_id"]
                services = dial["services"]
                utterances = ''
                states_record = {service: dict() for service in services}
                slot_positions = {service: dict() for service in services}
                bounds_record = defaultdict(list)   # global
                for turn in dial["turns"]:
                    utterances += turn["speaker"].strip().capitalize() + ": "
                    utterance = turn["utterance"]
                    if args.with_labels:
                        for frame in turn["frames"]:
                            service = frame["service"]
                            if service not in states_record:
                                continue
                            for span_chunk in frame["slots"]:
                                slot = span_chunk["slot"]
                                if "copy_from" in span_chunk:
                                    continue
                                start = span_chunk["start"]
                                ex_end = span_chunk["exclusive_end"]
                                value = utterance[start: ex_end]
                                bias = len(utterances)
                                bounds_record[value.lower()].append((bias + start, bias + ex_end))
                            if "state" in frame:
                                curr_state = frame["state"]["slot_values"]
                                updated_slots = get_state_updates(states_record[service], curr_state)
                                for slot in updated_slots:
                                    slot_positions[service][slot] = len(utterance) // 2 + len(utterances)
                                states_record[service] = curr_state
                    utterances += utterance + ' '
                 
                for service in services:
                    service_desc = noncat_descriptions[service]["service_desc"]
                    if args.with_labels:
                        states = states_record[service]
                    for slot in sorted(noncat_descriptions[service]["slot_descs"].keys()):
                        slot_desc = noncat_descriptions[service]["slot_descs"][slot]
                        if args.with_labels:
                            values = states.get(slot, [])
                            if len(values) > 0:
                                active, start, end = 1, -1, -1
                                value = values[0]
                                slot_position = slot_positions[service][slot]
                                for v in values:
                                    bounds = bounds_record.get(v.lower(), [])
                                    if len(bounds) > 0:
                                        start, end = min(bounds, key=lambda i: (abs((i[0] + i[1]) // 2 - slot_position), i))
                                        value = v
                                        break
                                    elif v == "dontcare":
                                        start, end = slot_position, slot_position + 1
                                        value = v
                                        break
                                    else:
                                        continue
                                
                                if (utterances[start: end].lower() != value.lower() and "dontcare" != value):
                                    print("Not matched: {} | {} | {} | {} | {}".format(fn, dial_id, service, slot, value))
                                    if not args.keep_no_matched:
                                        continue
                            else:
                                active, start, end = 0, -1, -1
                                value = ''
                        else:
                            active, start, end = -1, -1, -1
                            value = ''

                        data.append({"id": len(data),
                                    "dial_id": dial_id,
                                    "utterances": utterances,
                                    "service": service,
                                    "service_desc": service_desc,
                                    "slot": slot,
                                    "slot_desc": slot_desc,
                                    "active": active,
                                    "start": start,
                                    "end": end,
                                    "value": value})

    ans_count = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    with open(args.out_file, 'w') as wf:
        for d in data:
            print(json.dumps(d), file=wf)
            ans_count += d["active"]
    print(ans_count / len(data))
