export CUDA_VISIBLE_DEVICES='2'

#python ../CatChoice_WD/make_data.py -d ../datasets/data/dev -s ../datasets/data/schema.json -o cat_task_data/dev.jsonl
#python ../CatChoice_WD/predict.py --test_file cat_task_data/dev.jsonl --target_dir ../CatChoice_WD/saved/large_align_val --out_file outputs/dev/cat_results.json
#python ../NoncatSpan/make_data.py -d ../datasets/data/dev -s ../datasets/data/schema.json -o noncat_task_data/dev.jsonl
#python ../NoncatSpan/predict.py --test_file noncat_task_data/dev.jsonl --target_dir ../NoncatSpan/saved/large_align_val --out_file outputs/dev/noncat_results.json
#python merge_to_csv.py --in_files outputs/dev/cat_results.json outputs/dev/noncat_results.json --out_json outputs/dev/results.json --out_csv outputs/dev/results.csv
#python eval.py -d ../datasets/data/dev -s ../datasets/data/schema.json  -i outputs/dev/results.json
#python eval.py -d ../datasets/data/dev -s ../datasets/data/schema.json  -i outputs/dev/results.json -c noncat
#python eval.py -d ../datasets/data/dev -s ../datasets/data/schema.json  -i outputs/dev/results.json -c cat
#python eval.py -d ../datasets/data/dev -s ../datasets/data/schema.json  -i outputs/dev/results.json -c cat_num
#python eval.py -d ../datasets/data/dev -s ../datasets/data/schema.json  -i outputs/dev/results.json -c cat_bool
#python eval.py -d ../datasets/data/dev -s ../datasets/data/schema.json  -i outputs/dev/results.json -c cat_text

#python ../CatChoice_WD/make_data.py -d ../datasets/data/test_seen -s ../datasets/data/schema.json -o cat_task_data/test_seen.jsonl --norm
#python ../CatChoice_WD/predict.py --test_file cat_task_data/test_seen.jsonl --target_dir ../CatChoice_WD/saved/large_align_aug --out_file outputs/test_seen_/cat_results.json
#python ../NoncatSpan/make_data.py -d ../datasets/data/test_seen -s ../datasets/data/schema.json -o noncat_task_data/test_seen.jsonl --norm
#python ../NoncatSpan/predict.py --test_file noncat_task_data/test_seen.jsonl --target_dir ../NoncatSpan/saved/large_align_aug --out_file outputs/test_seen_/noncat_results.json
#python merge_to_csv.py --in_files outputs/test_seen_/cat_results.json outputs/test_seen_/noncat_results.json --out_csv outputs/test_seen_/results.csv

#python ../CatChoice_WD/make_data.py -d ../datasets/data/test_unseen -s ../datasets/data/schema.json -o cat_task_data/test_unseen.jsonl --norm
#python ../CatChoice_WD/predict.py --test_file cat_task_data/test_unseen.jsonl --target_dir ../CatChoice_WD/saved/large_align_aug --out_file outputs/test_unseen/cat_results.json
#python ../NoncatSpan/make_data.py -d ../datasets/data/test_unseen -s ../datasets/data/schema.json -o noncat_task_data/test_unseen.jsonl --norm
#python ../NoncatSpan/predict.py --test_file noncat_task_data/test_unseen.jsonl --target_dir ../NoncatSpan/saved/large_align_aug --out_file outputs/test_unseen/noncat_results.json
#python merge_to_csv.py --in_files outputs/test_unseen/cat_results.json outputs/test_unseen/noncat_results.json --out_csv outputs/test_unseen/results.csv

