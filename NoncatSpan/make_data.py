import os
import json
import argparse
import glob
from collections import defaultdict
from easynmt import EasyNMT
import random

random.seed(1114)


def get_state_updates(prev_state, curr_state):
    state_updates = dict(curr_state)
    for slot, values in curr_state.items():
        if slot in prev_state and sorted(prev_state[slot]) == sorted(values):
            del state_updates[slot]

    return state_updates

def back_trans(texts, translator, mid_lang):
    if mid_lang == "en":
        return texts

    texts = translator.translate(texts, source_lang="en", target_lang=mid_lang, batch_size=64)
    texts = translator.translate(texts, source_lang=mid_lang, target_lang="en", batch_size=64)
    
    return texts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dial_dirs", nargs='+', required=True, type=str)
    parser.add_argument("-s", "--schema_file", required=True, type=str)
    parser.add_argument("-o", "--out_file", required=True, type=str)
    parser.add_argument("-l", "--with_labels", action="store_true")
    parser.add_argument("--keep_no_matched", action="store_true")
    parser.add_argument("-sp", "--shuffle_prob", default=0, type=float)
    parser.add_argument("-a", "--aug_prob", default=0, type=float)
    parser.add_argument("-n", "--norm", action="store_true")
    args = parser.parse_args()

    with open(args.schema_file, 'r') as rf:
        schema = json.load(rf)
    if args.aug_prob > 0:
        translator = EasyNMT("m2m_100_418M")   # or: EasyNMT('m2m_100_1.2B') 
        lang_list = ["en", "es", "de", "ru", "zh"]
    else:
        translator = None
        lang_list = ["en"]
    
    noncat_descriptions = defaultdict(dict)
    ori_descs = []
    desc_mappings = []
    for service_i, service_chunk in enumerate(schema):
        service = service_chunk["service_name"]
        ori_descs.append(service_chunk["description"])
        desc_mappings.append((service, ))
        noncat_descriptions[service]["slot_descs"] = dict()
        for slot_i, slot_chunk in enumerate(service_chunk["slots"]):
            if slot_chunk["is_categorical"]:
                continue
            slot = slot_chunk["name"]
            ori_descs.append(slot_chunk["description"])
            desc_mappings.append((service, slot))

    all_descs = zip(*[back_trans(ori_descs, translator, lang) for lang in lang_list])
    for descs, desc_map in zip(all_descs, desc_mappings):
        if len(desc_map) == 1:
            if args.norm:
                noncat_descriptions[desc_map[0]]["service_descs"] = [desc.capitalize() for desc in descs]
            else:
                noncat_descriptions[desc_map[0]]["service_descs"] = descs
        else:
            if args.norm:
                noncat_descriptions[desc_map[0]]["slot_descs"][desc_map[1]] = [desc.capitalize() for desc in descs]
            else:
                noncat_descriptions[desc_map[0]]["slot_descs"][desc_map[1]] = descs
    
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
                    if "service_descs" in noncat_descriptions[service]:
                        if random.random() < args.aug_prob:
                            service_desc = random.choice(noncat_descriptions[service]["service_descs"][1:])
                        else:
                            service_desc = noncat_descriptions[service]["service_descs"][0]
                    if args.with_labels:
                        states = states_record[service]
                    for slot in sorted(noncat_descriptions[service]["slot_descs"].keys()):
                        if random.random() < args.aug_prob:
                            slot_desc = random.choice(noncat_descriptions[service]["slot_descs"][slot][1:])
                        else:
                            slot_desc = noncat_descriptions[service]["slot_descs"][slot][0]
                        if args.with_labels:
                            values = [v.lower() for v in states.get(slot, [])]
                            if len(values) > 0:
                                active, start, end = 1, -1, -1
                                if random.random() < args.shuffle_prob:
                                    random.shuffle(values)
                                value = values[0]
                                slot_position = slot_positions[service][slot]
                                for v in values:
                                    bounds = bounds_record.get(v, [])
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
                                
                                if (utterances[start: end].lower() != value and "dontcare" != value):
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
