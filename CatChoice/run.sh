export CUDA_VISIBLE_DEVICES='1'
python make_data.py -d ../datasets/data/new-train -s ../datasets/data/schema.json -o task_data/train.jsonl
