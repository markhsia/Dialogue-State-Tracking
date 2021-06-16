import argparse
import json
from collections import defaultdict

def write_csv(ans, output_path):
    ans = sorted(ans.items(), key=lambda x: x[0])
    with open(output_path, 'w') as f:
        f.write('id,state\n')
        for dialogue_id, states in ans:
            if len(states) == 0:  # no state ?
                str_state = 'None'
            else:
                states = sorted(states.items(), key=lambda x: x[0])
                str_state = ''
                for slot, value in states:
                    # NOTE: slot = "{}-{}".format(service_name, slot_name)
                    str_state += "{}={}|".format(
                            slot.lower(), value.replace(',', '_').lower())
                str_state = str_state[:-1]
            f.write('{},{}\n'.format(dialogue_id, str_state))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_files", type=str, nargs='+', required=True)
    parser.add_argument("--out_file", type=str, default="./results.csv")
    args = parser.parse_args()
    
    all_results = defaultdict(dict)
    for in_file in args.in_files:
        with open(in_file, 'r') as f:
            results = json.load(f)
        for k, v in results.items():
            all_results[k] = {**all_results[k], **v}
    
    write_csv(all_results, args.out_file)
