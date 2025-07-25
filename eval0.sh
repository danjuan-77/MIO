# !/bin/bash
export CUDA_VISIBLE_DEVICES=0

python eval.py --model_path /share/nlp/tuwenming/models/m-a-p/MIO-7B-Instruct \
    --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LAQA