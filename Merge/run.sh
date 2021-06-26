export CUDA_VISIBLE_DEVICES='1'

#python ../CatChoice/make_data.py -d ../datasets/data/dev -s ../datasets/data/schema.json -o cat_task_data/dev.jsonl
#python ../CatChoice/predict.py --test_file cat_task_data/dev.jsonl --target_dir ../CatChoice/saved/large_align_val --out_file outputs/dev/cat_results.json
#python ../NoncatSpan/make_data.py -d ../datasets/data/dev -s ../datasets/data/schema.json -o noncat_task_data/dev.jsonl
#python ../NoncatSpan/predict.py --test_file noncat_task_data/dev.jsonl --target_dir ../NoncatSpan/saved/large_align_val --out_file outputs/dev/noncat_results.json
#python merge_to_csv.py --in_files outputs/dev/cat_results.json outputs/dev/noncat_results.json --out_json outputs/dev/results.json --out_csv outputs/dev/results.csv
#python eval.py -d ../datasets/data/dev -s ../datasets/data/schema.json  -i outputs/dev_align_aug/results.json
#python eval.py -d ../datasets/data/dev -s ../datasets/data/schema.json  -i outputs/dev_align_aug/results.json -c noncat
#python eval.py -d ../datasets/data/dev -s ../datasets/data/schema.json  -i outputs/dev_align_aug/results.json -c cat
#python eval.py -d ../datasets/data/dev -s ../datasets/data/schema.json  -i outputs/dev_align_aug/results.json -c cat_num
#python eval.py -d ../datasets/data/dev -s ../datasets/data/schema.json  -i outputs/dev_align_aug/results.json -c cat_bool
#python eval.py -d ../datasets/data/dev -s ../datasets/data/schema.json  -i outputs/dev_align_aug/results.json -c cat_text

#python ../CatChoice/make_data.py -d ../datasets/data/test_seen -s ../datasets/data/schema.json -o cat_task_data/test_seen.jsonl
#python ../CatChoice/predict.py --test_file cat_task_data/test_seen.jsonl --target_dir ../CatChoice/saved/large_align_aug --out_file outputs/test_seen/cat_results.json
#python ../NoncatSpan/make_data.py -d ../datasets/data/test_seen -s ../datasets/data/schema.json -o noncat_task_data/test_seen.jsonl
#python ../NoncatSpan/predict.py --test_file noncat_task_data/test_seen.jsonl --target_dir ../NoncatSpan/saved/large_align_aug --out_file outputs/test_seen/noncat_results.json
#python merge_to_csv.py --in_files outputs/test_seen/cat_results.json outputs/test_seen/noncat_results.json --out_csv outputs/test_seen/results.csv

python ../CatChoice/make_data.py -d ../datasets/data/test_unseen -s ../datasets/data/schema.json -o cat_task_data/test_unseen.jsonl
python ../CatChoice/predict.py --test_file cat_task_data/test_unseen.jsonl --target_dir ../CatChoice/saved/large_align_aug --out_file outputs/test_unseen/cat_results.json
python ../NoncatSpan/make_data.py -d ../datasets/data/test_unseen -s ../datasets/data/schema.json -o noncat_task_data/test_unseen.jsonl
python ../NoncatSpan/predict.py --test_file noncat_task_data/test_unseen.jsonl --target_dir ../NoncatSpan/saved/0625-1536 --out_file outputs/test_unseen/noncat_results.json
python merge_to_csv.py --in_files outputs/test_unseen/cat_results.json outputs/test_unseen/noncat_results.json --out_csv outputs/test_unseen/results.csv

