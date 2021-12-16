# Dialogue-State-Tracking

Final Project of Applied Deep Learning 2021 at NTU (Lectured by Yun-Nung / Vivian, Chen)

## Task Description

Extract values for each task slot from mission-oriented dialogues.
* [Slides for task description](https://docs.google.com/presentation/d/1vekovUzNlffmbTyM4X3auGHt2P5PKUfDV2_eea5ycAU/edit#slide=id.p)
* [Kaggle for seen domain](https://www.kaggle.com/c/adl-final-dst-with-chit-chat-seen-domains) (Ranked 1st out of 22 teams)
* [Kaggle for unseen domain](https://www.kaggle.com/c/adl-final-dst-with-chit-chat-unseen-domains) (Ranked 3rd out of 22 teams)

## Report

https://github.com/joe0123/Dialogue-State-Tracking/blob/master/report.pdf

## Presentation Slides

https://docs.google.com/presentation/d/102wH0nR3VlnJXKe-q3dj0A1LmvNprzFMlDjqUU_GrCI/edit#slide=id.p


## Environment

Conda is required. If you haven't installed it, please refer to [documentation](https://docs.conda.io/en/latest/miniconda.html).

Then, please create the environment with the following commands:

```
cd ./env/
bash create.sh
conda activate dst
```

## Training

Three models are required to reproduce our best results in Kaggle. If you want to train them by yourself, please refer to the following commands.

### Span-DST

This model is used for non-categorical slots in both seen and unseen task.

```
cd ./NoncatSpan/
python make_data.py -d [train data dir path] [dev data dir path] -s [schema path] -o task_data/all.jsonl -l -a 0.6 --norm
python train.py --train_file task_data/all.jsonl --model_name xlnet-large-cased
```

### Choice-DST (deep)

This model is used for categorical slots in seen task.

```
cd ./CatChoice/
python make_data.py -d [train data dir path] [dev data dir path] -s [schema path] -o task_data/all.jsonl -l -a 0.6 --norm
python train.py --train_file task_data/all.jsonl --model_name roberta-large
```

### Choice-DST (deep and wide)

This model is used for categorical slots in unseen task.

```
cd ./CatChoice_WD/
python make_data.py -d [train data dir path] [dev data dir path] -s [schema path] -o task_data/all.jsonl -l -a 0.6 --norm
python train.py --train_file task_data/all.jsonl --model_name roberta-large
```


## Inference

You can reproduce the best results in Kaggle with the following commands.

First, you should download and extract our model checkpoints.
```
python download.py
```

Then, just run the commands below for reproduction. Note that the data directory must contain *test_seen/*, *test_unseen/* and *schema.json*.
```
cd ./Merge/
bash run.sh [data dir path]
```

Finally, *./Merge/outputs/test_seen/results.csv* and *./Merge/outputs/test_unseen/results.csv* are the submission files for seen task and unseen task, respectively.
