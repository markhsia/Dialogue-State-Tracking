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
            # Remove the slot from state if its value didn't change.
            state_updates.pop(slot)

    return state_updates


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
            value_bounds = {"dontcare": (-1, -1)}
            prev_states = dict()
            records = defaultdict(dict)
            with_labels = False
            for turn in dial["turns"]:
                utterances += turn["speaker"].strip().capitalize() + ": "
                utterance = turn["utterance"]
                if "frames" in turn:
                    with_labels = True
                    for frame in turn["frames"]:
                        service = frame["service"]
                        update_bounds = False
                        for span_chunk in frame["slots"]:
                            if "copy_from" in span_chunk:
                                continue
                            start = span_chunk["start"]
                            ex_end = span_chunk["exclusive_end"]
                            value = utterance[start: ex_end]
                            bias = len(utterances)
                            update_bounds = True
                            value_bounds[value] = (bias + start, bias + ex_end)
                            value_bounds["MWOZ__" + value.lower()] = (bias + start, bias + ex_end)
                        if update_bounds and "state" in frame:
                            curr_state = frame["state"]["slot_values"]
                            for slot, values in curr_state.items():
                                bounds = max([value_bounds.get(v, (-1, -1)) for v in values] + \
                                            [value_bounds.get("MWOZ__" + v, (-1, -1)) for v in values])
                                records[service][slot] = {"bounds": bounds, "values": values}
                            prev_states[service] = curr_state
                utterances += utterance + ' '
             
            if with_labels:
                for service in services:
                    service_desc = noncat_descriptions[service]["service_desc"]
                    for slot in sorted(noncat_descriptions[service]["slot_descs"].keys()):
                        slot_desc = noncat_descriptions[service]["slot_descs"][slot]
                        record = records[service].get(slot)
                        if record != None:
                            start, end = record["bounds"]
                            values = record["values"]
                            lower_values = [v.lower() for v in values]
                            if (utterances[start: end].lower() not in lower_values and "dontcare" not in lower_values):
                                print("Not matched: {} | {} | {} | {}".format(fn, dial_id, slot, values))
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
