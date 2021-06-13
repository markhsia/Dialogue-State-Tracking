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
    parser.add_argument("-o", "--out_file", required=True, type=str)
    parser.add_argument("--keep_no_matched", action="store_true")
    args = parser.parse_args()

    with open(args.schema_file, 'r') as rf:
        schema = json.load(rf)
    cat_descriptions = defaultdict(dict)
    cat_slots = defaultdict(set)
    cat_poss_values = defaultdict(dict)
    cat_poss_value_ids = defaultdict(lambda: defaultdict(dict))
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
            for v_i, v in enumerate(poss_values):
                cat_poss_value_ids[service][slot][v] = v_i

    data = []
    for fn in sorted(glob.glob(os.path.join(args.dial_dir, "dialogues_*.json"))):
        with open(fn, 'r') as rf:
            dials = json.load(rf)
        
        for dial in dials:
            dial_id = dial["dialogue_id"]
            services = dial["services"]
            utterances = ''
            states_record = {service: dict() for service in services}
            bounds_record = {service: defaultdict(dict) for service in services}
            with_labels = False
            for turn in dial["turns"]:
                utterances += turn["speaker"].strip().capitalize() + ": "
                utterance = turn["utterance"]
                if "frames" in turn:
                    with_labels = True
                    for frame in turn["frames"]:
                        service = frame["service"]
                        if service not in states_record:
                            continue
                        for act_chunk in frame["actions"]:
                            slot = act_chunk["slot"]
                            if slot not in cat_slots[service]:
                                slot = "{}-{}".format(service, slot)
                            if slot not in cat_slots[service]:
                                continue
                            if "values" in act_chunk:
                                values = act_chunk["values"]
                            else:
                                values = [act_chunk["value"]]
                            for value in values:
                                bounds_record[service][slot][value] = (len(utterances), len(utterances) + len(utterance))
                        if "state" in frame:
                            states_record[service] = frame["state"]["slot_values"]
                utterances += utterance + ' '

            if with_labels:
                for service in services:
                    service_desc = cat_descriptions[service]["service_desc"]
                    states = states_record[service]
                    bounds = bounds_record[service]
                    for slot, poss_values in sorted(cat_poss_values[service].items()):
                        slot_desc = cat_descriptions[service]["slot_descs"][slot]
                        values = states.get(slot, ["unknown"])
                        (start, end), value = max([(bounds[slot].get(v, (-1, -1)), v) for v in values])
                        if (start, end) == (-1, -1) and value != "unknown":
                            print(bounds)
                            print("Not matched: {} | {} | {} | {} | {}".format(fn, dial_id, service, slot, values))
                        data.append({"id": len(data),
                                    "dial_id": dial_id,
                                    "utterances": utterances,
                                    "service": service,
                                    "service_desc": service_desc,
                                    "slot": slot,
                                    "slot_desc": slot_desc,
                                    "poss_values": poss_values,
                                    "label": cat_poss_value_ids[service][slot][value],
                                    "start": start,
                                    "end": end})
                    
            else:
                raise NotImplementedError
    
    known_count = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    with open(args.out_file, 'w') as wf:
        for d in data:
            print(json.dumps(d), file=wf)
            known_count += int(d["label"] != 0)
    print(known_count / len(data))
