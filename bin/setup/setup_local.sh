source env/bin/activate
export CUDA_VISIBLE_DEVICES=$1
export REM_ROOT=`pwd`
export PYTHONPATH=$REM_ROOT:$PYTHONPATH
export PYTHON_EXEC=python
export HF_DATASETS_CACHE=~/Workspace/hf_datasets
export TRANSFORMERS_CACHE=~/Workspace/hf_models