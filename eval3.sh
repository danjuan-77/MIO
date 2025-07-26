# !/bin/bash
export CUDA_VISIBLE_DEVICES=3


python eval.py --model_path /share/nlp/tuwenming/models/m-a-p/MIO-7B-Instruct --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVL
python eval.py --model_path /share/nlp/tuwenming/models/m-a-p/MIO-7B-Instruct --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVM

# nohup bash eval3.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_mio_gpu3_$(date +%Y%m%d%H%M%S).log 2>&1 &