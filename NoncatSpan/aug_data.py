import argparse
import json
import random
from easynmt import EasyNMT

def back_translation(text, translator, mid_lang):
    text = translator.translate(text, source_lang="en", target_lang=mid_lang)
    text = translator.translate(text, source_lang=mid_lang, target_lang="en")

    return text

def back_translation(text, translator, mid_lang):
    if mid_lang == "en":
        return text

    text = translator.translate(text, source_lang="en", target_lang=mid_lang)
    text = translator.translate(text, source_lang=mid_lang, target_lang="en")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_file", required=True, type=str)
    parser.add_argument("-s", "--schema_file", required=True, type=str)
    parser.add_argument("-o", "--out_file", required=True, type=str)
    parser.add_argument("-a", "--aug_ratio", default=0.5)
    parser.add_argument("--seed", default=1114)
    args = parser.parse_args()

    random.seed(args.seed)
    translator = EasyNMT("m2m_100_418M")   #or: EasyNMT('m2m_100_1.2B') 
    lang_list = ["es", "fr", "de", "pt", "ru"]


    with open(args.in_file, 'r') as rf:#, open(args.out_file, 'w') as wf:
        for line in rf:
            data = json.loads(line)
            if random.random() < args.aug_ratio:
                mid_lang = random.choice(lang_list)
                print(data["service_desc"])
                data["service_desc"] = back_translation(data["service_desc"], translator, mid_lang)
                print(mid_lang, data["service_desc"])
                print('\n', flush=True)
            if random.random() < args.aug_ratio:
                mid_lang = random.choice(lang_list)
                print(data["slot_desc"])
                data["slot_desc"] = back_translation(data["slot_desc"], translator, mid_lang)
                print(mid_lang, data["slot_desc"])
                print('\n', flush=True)
