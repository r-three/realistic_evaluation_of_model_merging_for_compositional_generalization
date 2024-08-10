# Realistic Evaluation of Model Merging Methods

This contains the code computing the statistics and doing inference for Realistic Evaluation of Model Merging Methods. 
We use `git-theta` to compute the merges, which allows for merging each parameter block indepedently, which makes it possible to merge large models on smaller. 

## Setup

1. Create a virtual environment and activate it.
```
python3.10 -m venv env
source env/bin/activate
```
2. Install dependencies 
```
python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

A modified version of Promptsource must be installed from source. 
```
cd promptsource 
python -m pip install -e . 
```
Download multilingual ROUGE scorer from `https://github.com/csebuetnlp/xl-sum/tree/master/multilingual_rouge_scoring`
```
git clone https://github.com/csebuetnlp/xl-sum.git
cd multilingual_rouge_scoring
mv io.py io_original.py
python -m pip install -r requirements.txt
python -m pip install .
mv io_original.py io.py 
```
A modified version of open_clip must be installed from source. 
```
cd open_clip
python -m pip install -e . 
```

1. Set environment variables (This step has to be done every session)
```
source bin/setup_{machine}.sh
```
## Saving Statistics 

An example of how save the statistics for TIES, RegMean, Fisher Merging, and MaTS are shown below. 
Note that though TIES has not statistics, we treat the trimmed model as a statistic so that the merge can be computed for each parameter block indepedently. 


Trimmed model 
```
python src/save_trimmed_model.py --pretrained_model_name {pretrained_model_name} --checkpoint_filepath {checkpoint_filepath} --save_path_for_trimmed_model {path_to_save_model}
```

RegMean and MaTS
```
python src/save_gram_matrices.py    -c configs/evaluation_run/language.json  configs/evaluation_dataset/p3.json  configs/model/mt5_xl_lm_adapt.json  configs/model/full_model.json   -er eval_batch_size=2 -ed dataset=squad  split=train -m filepath_to_load_model="exp_out/p3/squad/models-google-mt5-xl-lm-adapt/full_model/2024-02-06-21-15-12/checkpoints/best_checkpoint_199.pt" --output_path gram_squad.pt
```

Fisher Merging 
```
python src/save_fisher.py     -c configs/evaluation_run/language.json  configs/evaluation_dataset/p3.json  configs/model/mt5_xl_lm_adapt.json  configs/model/full_model.json    -ed dataset=squad split=train -m filepath_to_load_model="exp_out/p3/squad/models-google-mt5-xl-lm-adapt/full_model/2024-02-06-21-15-12/checkpoints/best_checkpoint_199.pt" --output_path fisher_squad.pt
```

## Training 

Run the `training` script with 

`-c` the list of configs for the model

`-td` any training dataset config parameters to update. 

`-ed` any evaluation dataset config parameters to update. 

`-tr` any training run config parameters to update. 

`-er` any evaluation run config parameters to update. 

`-m` any model config parameters to update

Cross Lingual: 
```
python src/train/training.py -c configs/model/mt5_xl_lm_adapt.json configs/model/full_model.json  configs/training_run/cross_lingual.json     configs/training_dataset/p3.json  configs/evaluation_dataset/p3.json configs/evaluation_run/language.json  -tr micro_train_batch_size=2  train_dataset_mixture=multitask_multilingual -er eval_batch_size=4  dataset_mixture=multitask_multilingual
```

```
python src/train/training.py -c configs/model/mt5_xl_lm_adapt.json configs/model/full_model.json  configs/training_run/cross_lingual.json     configs/training_dataset/p3.json  configs/evaluation_dataset/p3.json configs/evaluation_run/language.json -td dataset=squad  -ed dataset=squad -tr micro_train_batch_size=2 -er eval_batch_size=4
```

```
python src/train/training.py -c configs/model/mt5_xl_lm_adapt.json configs/model/full_model.json  configs/training_run/cross_lingual.json     configs/training_dataset/p3.json  configs/evaluation_dataset/p3.json configs/evaluation_run/language.json -td dataset=xnli language_code=ar  -ed dataset=xnli language_code=ar -tr micro_train_batch_size=2 -er eval_batch_size=32
```

```
python src/train/training.py -c configs/model/mt5_xl_lm_adapt.json  configs/model/full_model.json configs/training_run/cross_lingual.json     configs/training_dataset/p3.json  configs/evaluation_dataset/p3.json configs/evaluation_run/language.json -td dataset=wiki_lingua language=thai  -ed dataset=wiki_lingua language=thai -tr micro_train_batch_size=1 -er eval_batch_size=4
```

```
python src/train/training.py -c configs/model/mt5_xl_lm_adapt.json  configs/model/full_model.json configs/training_run/cross_lingual.json     configs/training_dataset/p3.json  configs/evaluation_dataset/p3.json configs/evaluation_run/language.json  -td dataset=tydiqa language=korean  -ed dataset=tydiqa language=korean -tr micro_train_batch_size=2 -er eval_batch_size=4
```

```
python src/train/training.py -c configs/model/mt5_xl_lm_adapt.json  configs/model/full_model.json configs/training_run/cross_lingual.json     configs/training_dataset/p3.json  configs/evaluation_dataset/p3.json configs/evaluation_run/language.json  -td dataset=xlwic language_code=de  -ed dataset=xlwic language_code=de -tr micro_train_batch_size=4 -er eval_batch_size=16
```

DomainNet 

One Domain 
```
python src/train/training.py -c configs/model/clip.json configs/training_run/domainnet.json configs/training_dataset/domainnet.json configs/evaluation_dataset/domainnet_training.json configs/evaluation_run/vision.json -td domain=clipart task=mammal -ed domain=clipart task=mammal -tr micro_train_batch_size=128 -er eval_batch_size=256
```
All Domains
```
python src/train/training.py -c configs/model/clip.json configs/training_run/domainnet.json configs/training_dataset/domainnet_all.json configs/evaluation_dataset/domainnet_all.json configs/evaluation_run/vision.json -tr micro_train_batch_size=128 train_dataset_mixture=domainnet num_batches=10000 should_eval_before_training=False -er eval_batch_size=256 dataset_mixture=domainnet
```


## Saving Statistics 
Some methods (TIES, RegMean, Fisher Merging, and MaTS) require saving some statistics first. 
Since the merge is computed indedently for each parameter block, the trimmed model is the statistic for TIES. 


### DomainNet  
```
python src/merging/save_statistic.py    -c configs/evaluation_run/vision.json  configs/evaluation_dataset/domainnet.json  configs/model/clip.json  configs/model/full_model.json  configs/merging/domainnet.json  configs/merging/{method}.json -er eval_batch_size=32
```

### Cross Lingual 
```
python src/merging/save_statistic.py    -c configs/evaluation_run/language.json  configs/evaluation_dataset/p3.json  configs/model/mt5_xl_lm_adapt.json  configs/model/full_model.json  configs/merging/multitask_multilingual.json  configs/merging/{method}.json -er eval_batch_size=32

```

## Merging

We use `git-theta` to compute the merge and recommend and creating a separate repo for tracking the models to not tangle the code and model `.git`

To do so, first clone the repo. To not tangle the `.git`, we recommend cloning `git-theta` is a different directory not under this one.  
``` 
git clone https://github.com/blester125/git-theta
git checkout feat/merge-cli 
python -m pip install -e . 
```

We also recommend creating a new git repo for tracking models not under this one to not tangle the `.git`
```
mkdir merged-models
mv ../exp_out . 
git init 
git theta track 
```

Follow the instructions at `https://github.com/blester125/git-theta/tree/feat/merge-cli/plugins/merge_cli` to start using `git-theta`. 


### Evaluation 

Run the `inference` script with

`-e` path to experiment dir of model 

`--merged_model` path to merged model 


Inference on best checkpoint from experiment. The correct dataset and model config are noted from the experiment path.  
```
python src/eval/inference.py -e {exp_dir}
```

Inference on merged model. The correct evaluation config, evaluation dataset config, and model configs must be passed in. 
```
python src/eval/inference.py -c configs/model/mt5_xl_lm_adapt.json  configs/evaluation_dataset/p3.json configs/evaluation_run/language.json  --merged_model average.pt --output_dir average  
```

### Released Checkpoints 
The domainnet and cross lingual checkpoints can be found here: 
We also include a Pytorch version of mT5-xl-lm-adapt already converted from the default Jax format. 

## Citation

If you find this repo helpful, feel free to cite our work:

