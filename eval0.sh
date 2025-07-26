# !/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Level 1
python eval.py --model_path /share/nlp/tuwenming/models/m-a-p/MIO-7B-Instruct --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LAQA
python eval.py --model_path /share/nlp/tuwenming/models/m-a-p/MIO-7B-Instruct --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LIQA

# nohup bash eval0.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_mio_gpu0_$(date +%Y%m%d%H%M%S).log 2>&1 &