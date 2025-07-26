# !/bin/bash
export CUDA_VISIBLE_DEVICES=1

python eval.py --model_path /share/nlp/tuwenming/models/m-a-p/MIO-7B-Instruct --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/VAR

# Level 4
python eval.py --model_path /share/nlp/tuwenming/models/m-a-p/MIO-7B-Instruct --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVC

# nohup bash eval5.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_mio_gpu5_$(date +%Y%m%d%H%M%S).log 2>&1 &