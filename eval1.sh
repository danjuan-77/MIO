# !/bin/bash
export CUDA_VISIBLE_DEVICES=1

# Level 1
python eval.py --model_path /share/nlp/tuwenming/models/m-a-p/MIO-7B-Instruct --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LVQA

# Level 2
python eval.py --model_path /share/nlp/tuwenming/models/m-a-p/MIO-7B-Instruct --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_2/MAIC

# nohup bash eval1.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_mio_gpu1_$(date +%Y%m%d%H%M%S).log 2>&1 &