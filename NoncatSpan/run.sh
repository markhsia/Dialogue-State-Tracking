export CUDA_VISIBLE_DEVICES='0'
#python make_data.py -d ../datasets/data/train -s ../datasets/data/schema.json -o task_data/train.jsonl -l -a 0.6 --norm
#python make_data.py -d ../datasets/data/dev -s ../datasets/data/schema.json -o task_data/dev.jsonl -l --keep_no_matched --norm
#python train.py --train_file task_data/train.jsonl --valid_file task_data/dev.jsonl --model_name xlnet-large-cased --epoch 5

#python make_data.py -d ../datasets/data/train ../datasets/data/dev -s ../datasets/data/schema.json -o task_data/all.jsonl -l -a 0.6
#python train.py --train_file task_data/all.jsonl --model_name xlnet-large-cased


python make_data.py -d ../datasets/data/dev -s ../datasets/data/schema.json -o task_data/dev.jsonl -l --keep_no_matched --norm
python eval.py --valid_file task_data/dev.jsonl --target_dir saved/0629-1704
