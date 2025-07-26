# !/bin/bash
export CUDA_VISIBLE_DEVICES=0

python eval.py --model_path /share/nlp/tuwenming/models/m-a-p/MIO-7B-Instruct --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVR
python eval.py --model_path /share/nlp/tuwenming/models/m-a-p/MIO-7B-Instruct --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/VAH

# nohup bash eval4.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_mio_gpu4_$(date +%Y%m%d%H%M%S).log 2>&1 &