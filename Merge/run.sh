export CUDA_VISIBLE_DEVICES='2'

python ../CatChoice/make_data.py -d ../datasets/data/dev -s ../datasets/data/schema.json -o cat_task_data/dev.jsonl
python ../CatChoice/predict.py --test_file cat_task_data/dev.jsonl --target_dir ../CatChoice/saved/large_val --out_file cat_results.json
#python ../NoncatSpan/make_data.py -d ../datasets/data/dev -s ../datasets/data/schema.json -o noncat_task_data/dev.jsonl
#python ../NoncatSpan/predict.py --test_file noncat_task_data/dev.jsonl --target_dir ../NoncatSpan/saved/0619-0916 --out_file noncat_results.json
python merge_to_csv.py --in_files cat_results.json noncat_results.json --out_file results.csv
python eval.py -d ../datasets/data/dev -i results.csv

#python ../CatChoice/make_data.py -d ../datasets/data/test_seen -s ../datasets/data/schema.json -o cat_task_data/test_seen.jsonl
#python ../CatChoice/predict.py --test_file cat_task_data/test_seen.jsonl --target_dir ../CatChoice/saved/large --out_file cat_results.json
#python ../NoncatSpan/make_data.py -d ../datasets/data/test_seen -s ../datasets/data/schema.json -o noncat_task_data/test_seen.jsonl
#python ../NoncatSpan/predict.py --test_file noncat_task_data/test_seen.jsonl --target_dir ../NoncatSpan/saved/large --out_file noncat_results.json
#python merge_to_csv.py --in_files cat_results.json noncat_results.json --out_file results.csv

#python ../CatChoice/make_data.py -d ../datasets/data/test_unseen -s ../datasets/data/schema.json -o cat_task_data/test_unseen.jsonl
#python ../CatChoice/predict.py --test_file cat_task_data/test_unseen.jsonl --target_dir ../CatChoice/saved/large --out_file cat_results.json
#python ../NoncatSpan/make_data.py -d ../datasets/data/test_unseen -s ../datasets/data/schema.json -o noncat_task_data/test_unseen.jsonl
#python ../NoncatSpan/predict.py --test_file noncat_task_data/test_unseen.jsonl --target_dir ../NoncatSpan/saved/large --out_file noncat_results.json
#python merge_to_csv.py --in_files cat_results.json noncat_results.json --out_file results.csv

