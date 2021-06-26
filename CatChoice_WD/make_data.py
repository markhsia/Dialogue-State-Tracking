import os
import json
import argparse
import glob
from collections import defaultdict, Counter
from easynmt import EasyNMT
import string
from num2words import num2words
import nltk
nltk.download("wordnet")
from nltk.corpus import wordnet
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

def create_wide_features(dial_id, service_num, slot_type, slot_type_count, utterances, slot, value):
    utterances = set(utterances.lower().translate(str.maketrans('', '', string.punctuation)).split())
    features = [int('_' not in dial_id), int(service_num > 1), int(slot_type_count > 1)]
    if slot_type == "num":
        features += [1, 0, 0]
        if value in ["unknown", "dontcare"]:
            features += [0, 0, 0]
        else:
            value_syn = num2words(value, to="cardinal")
            features += [1, int(value.lower() in utterances), int(value_syn.lower() in utterances)]
    elif slot_type == "bool":
        features += [0, 1, 0]
        if value in ["unknown", "dontcare"]:
            features += [0, 0, 0]
        else:
            slot_words = slot.split('_')
            slot_words = [s[0] for s in nltk.pos_tag(slot_words) if s[1][:2] in ["NN", "JJ"]]
            slot_syns = [l.name().replace('_', ' ') \
                        for w in slot_words for syn_set in wordnet.synsets(w) for l in syn_set.lemmas()]
            features += [1, int(any([s.lower() in utterances for s in slot_words])), \
                        int(any([s.lower() in utterances for s in slot_syns]))]
    elif slot_type == "text":
        features += [0, 0, 1]
        if value in ["unknown", "dontcare"]:
            features += [0, 0, 0]
        else:
            value_syns = [l.name().replace('_', ' ') for syn_set in wordnet.synsets(value) for l in syn_set.lemmas()]
            features += [1, int(value.lower() in utterances), \
                        int(any([v.lower() in utterances for v in value_syns]))]
    else:
        raise NotImplementedError
    
    assert len(features) == 9

    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dial_dirs", nargs='+', required=True, type=str)
    parser.add_argument("-s", "--schema_file", required=True, type=str)
    parser.add_argument("-o", "--out_file", required=True, type=str)
    parser.add_argument("-l", "--with_labels", action="store_true")
    parser.add_argument("-sp", "--shuffle_prob", default=0, type=float)
    parser.add_argument("-a", "--aug_prob", default=0, type=float)
    args = parser.parse_args()

    with open(args.schema_file, 'r') as rf:
        schema = json.load(rf)
    if args.aug_prob > 0:
        translator = EasyNMT("m2m_100_418M")   # or: EasyNMT('m2m_100_1.2B') 
        lang_list = ["en", "es", "de", "ru", "zh"]
    else:
        translator = None
        lang_list = ["en"]
    
    cat_descriptions = defaultdict(dict)
    cat_slots = defaultdict(set)
    slot_types = defaultdict(dict)
    slot_type_counts = defaultdict(Counter)
    cat_poss_values = defaultdict(dict)
    ori_descs = []
    desc_mappings = []
    for service_i, service_chunk in enumerate(schema):
        service = service_chunk["service_name"]
        ori_descs.append(service_chunk["description"])
        desc_mappings.append((service, ))
        cat_descriptions[service]["slot_descs"] = dict()
        for slot_i, slot_chunk in enumerate(service_chunk["slots"]):
            if not slot_chunk["is_categorical"]:
                continue
            slot = slot_chunk["name"]
            cat_slots[service].add(slot)
            ori_descs.append(slot_chunk["description"])
            desc_mappings.append((service, slot))
            poss_values = slot_chunk["possible_values"]
            #poss_values = [v.lower() for v in slot_chunk["possible_values"]]
            if all([v.isnumeric() for v in poss_values]):
                slot_types[service][slot] = "num"
                slot_type_counts[service]["num"] += 1
            elif sorted([v.lower() for v in poss_values]) == ["false", "true"]:
                slot_types[service][slot] = "bool"
                slot_type_counts[service]["bool"] += 1
            else:
                slot_types[service][slot]  = "text"
                slot_type_counts[service]["text"] += 1
            poss_values = ["unknown", "dontcare"] + poss_values
            cat_poss_values[service][slot] = poss_values
    
    all_descs = zip(*[back_trans(ori_descs, translator, lang) for lang in lang_list])
    for descs, desc_map in zip(all_descs, desc_mappings):
        if len(desc_map) == 1:
            cat_descriptions[desc_map[0]]["service_descs"] = descs
            #cat_descriptions[desc_map[0]]["service_descs"] = [desc.capitalize() for desc in descs]
        else:
            cat_descriptions[desc_map[0]]["slot_descs"][desc_map[1]] = descs
            #cat_descriptions[desc_map[0]]["slot_descs"][desc_map[1]] = [desc.capitalize() for desc in descs]


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
                                    bounds_record[service][slot][value.lower()] = (len(utterances), \
                                                                        len(utterances) + len(utterance))
                            if "state" in frame:
                                curr_state = frame["state"]["slot_values"]
                                state_updates = get_state_updates(states_record[service], curr_state)
                                for slot, values in state_updates.items():
                                    for value in values:
                                        if value not in bounds_record[service][slot]:
                                            bounds_record[service][slot][value.lower()] = (len(utterances), \
                                                                            len(utterances) + len(utterance))
                                states_record[service] = curr_state

                    utterances += utterance + ' '

                for service in services:
                    states = states_record[service]
                    for slot, poss_values in sorted(cat_poss_values[service].items()):
                        if args.with_labels:
                            true_values = [v.lower() for v in states.get(slot, ["unknown"])]
                            if random.random() < args.shuffle_prob:
                                random.shuffle(true_values)
                            true_value = true_values[0]
                        for poss_value in poss_values:
                            if random.random() < args.aug_prob:
                                service_desc = random.choice(cat_descriptions[service]["service_descs"][1:])
                            else:
                                service_desc = cat_descriptions[service]["service_descs"][0]
                            if random.random() < args.aug_prob:
                                slot_desc = random.choice(cat_descriptions[service]["slot_descs"][slot][1:])
                            else:
                                slot_desc = cat_descriptions[service]["slot_descs"][slot][0]
                            if args.with_labels:
                                if poss_value.lower() == true_value:
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
                            
                            slot_type = slot_types[service][slot]
                            wide_features = create_wide_features(dial_id, len(services), 
                                                                slot_type, slot_type_counts[service][slot_type],
                                                                utterances, slot, poss_value)
                            data.append({"id": total_slots_count,
                                        "dial_id": dial_id,
                                        "utterances": utterances,
                                        "wide_features": wide_features,
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
