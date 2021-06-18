import os
import json
import argparse
import glob
from collections import defaultdict
import random

random.seed(1114)

def get_dontcare_updates(prev_state, curr_state):
    slots = []
    for slot, values in curr_state.items():
        if "dontcare" in values and "dontcare" not in prev_state.get(slot, []):
            slots.append(slot)
    
    return slots


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
                global_bounds = {}
                states_record = {service: dict() for service in services}
                bounds_record = {service: dict() for service in services}
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
                                global_bounds[value] = (bias + start, bias + ex_end)
                                global_bounds["MWOZ__" + value.lower()] = (bias + start, bias + ex_end)
                            if "state" in frame:
                                curr_state = frame["state"]["slot_values"]
                                dontcare_slots = get_dontcare_updates(states_record[service], curr_state)
                                for slot in dontcare_slots:
                                    global_bounds["{}-{}-dontcare".format(service, slot)] = (len(utterances), \
                                                                                            len(utterances) + len(utterance))
                                states_record[service] = curr_state
                                bounds_record[service] = dict(global_bounds)
                    utterances += utterance + ' '
                 
                for service in services:
                    service_desc = noncat_descriptions[service]["service_desc"]
                    if args.with_labels:
                        states = states_record[service]
                        bounds = bounds_record[service]
                    for slot in sorted(noncat_descriptions[service]["slot_descs"].keys()):
                        slot_desc = noncat_descriptions[service]["slot_descs"][slot]
                        if args.with_labels:
                            values = states.get(slot, [])
                            if len(values) > 0:
                                active = 1
                                (start, end), value = max([(bounds.get(v, (-1, -1)), v) for v in values] + \
                                                [(bounds.get("MWOZ__" + v, (-1, -1)), v) for v in values if v.islower()] + \
                                                [(bounds.get("{}-{}-{}".format(service, slot, v)), v) \
                                                for v in values if v == "dontcare"])
                                lower_values = [v.lower() for v in values]
                                if (utterances[start: end].lower() not in lower_values and "dontcare" not in lower_values):
                                    #print(bounds)
                                    print("Not matched: {} | {} | {} | {} | {}".format(fn, dial_id, service, slot, values))
                                    if not args.keep_no_matched:
                                        continue
                            else:
                                active, start, end = 0, -1, -1
                                value, values = '', ['']
                        else:
                            active, start, end = -1, -1, -1
                            value, values = '', ['']

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
                                    "value": value,
                                    "values": values})
                        
    

    ans_count = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    with open(args.out_file, 'w') as wf:
        for d in data:
            print(json.dumps(d), file=wf)
            ans_count += d["active"]
    print(ans_count / len(data))
