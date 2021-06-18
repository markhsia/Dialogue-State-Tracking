export CUDA_VISIBLE_DEVICES='1'
python make_data.py -d ../datasets/data/train -s ../datasets/data/schema.json -o task_data/train.jsonl -l
python make_data.py -d ../datasets/data/dev -s ../datasets/data/schema.json -o task_data/dev.jsonl -l --keep_no_matched
python train.py --train_file task_data/train.jsonl --valid_file task_data/dev.jsonl --model_name xlnet-large-cased 

#python make_data.py -d ../datasets/data/train ../datasets/data/dev -s ../datasets/data/schema.json -o task_data/all.jsonl -l
#python train.py --train_file task_data/all.jsonl --model_name xlnet-large-cased

