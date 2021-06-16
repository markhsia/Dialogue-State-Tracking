export CUDA_VISIBLE_DEVICES='2'

#python ../CatChoice/make_data.py -d ../datasets/data/dev -s ../datasets/data/schema.json -o cat_task_data/dev.jsonl
#python ../CatChoice/predict.py --test_file cat_task_data/dev.jsonl --target_dir ../CatChoice/saved/0616-0556 --out_file cat_results.json

#python ../NonCatSpan/make_data.py -d ../datasets/data/dev -s ../datasets/data/schema.json -o noncat_task_data/dev.jsonl
#python ../NonCatSpan/predict.py --test_file noncat_task_data/dev.jsonl --target_dir ../NonCatSpan/saved/large_val --out_file noncat_results.json

#python merge_to_csv.py --in_files cat_results.json noncat_results.json --out_file results.csv
python eval.py -d ../datasets/data/dev -i results.csv
