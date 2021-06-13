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
    for fn in sorted(glob.glob(os.path.join(args.dial_dir, "dialogues_*.json"))):
        with open(fn, 'r') as rf:
            dials = json.load(rf)
        
        for dial in dials:
            dial_id = dial["dialogue_id"]
            services = dial["services"]
            utterances = ''
            global_bounds = {"dontcare": (-1, -1)}
            states_record = {service: dict() for service in services}
            bounds_record = {service: dict() for service in services}
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
                            states_record[service] = frame["state"]["slot_values"]
                            bounds_record[service] = dict(global_bounds)
                utterances += utterance + ' '
             
            if with_labels:
                for service in services:
                    service_desc = noncat_descriptions[service]["service_desc"]
                    states = states_record[service]
                    bounds = bounds_record[service]
                    for slot in sorted(noncat_descriptions[service]["slot_descs"].keys()):
                        slot_desc = noncat_descriptions[service]["slot_descs"][slot]
                        values = states.get(slot, [])
                        if len(values) > 0:
                            start, end = max([bounds.get(v, (-1, -1)) for v in values] + \
                                        [bounds.get("MWOZ__" + v, (-1, -1)) for v in values])
                            lower_values = [v.lower() for v in values]
                            if (utterances[start: end].lower() not in lower_values and "dontcare" not in lower_values):
                                print("Not matched: {} | {} | {} | {} | {}".format(fn, dial_id, service, slot, values))
                                if not args.keep_no_matched:
                                    continue
                            data.append({"id": len(data),
                                        "dial_id": dial_id,
                                        "utterances": utterances,
                                        "service": service,
                                        "service_desc": service_desc,
                                        "slot": slot,
                                        "slot_desc": slot_desc,
                                        "active": 1,
                                        "start": start,
                                        "end": end,
                                        "values": values})
                        
                        else:
                            data.append({"id": len(data),
                                        "dial_id": dial_id,
                                        "utterances": utterances,
                                        "service": service,
                                        "service_desc": service_desc,
                                        "slot": slot,
                                        "slot_desc": slot_desc,
                                        "active": 0,
                                        "start": -1,
                                        "end": -1,
                                        "values": ['']})
            else:
                raise NotImplementedError
    
    ans_count = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    with open(args.out_file, 'w') as wf:
        for d in data:
            print(json.dumps(d), file=wf)
            ans_count += d["active"]
    print(ans_count / len(data))
