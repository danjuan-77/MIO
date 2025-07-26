# !/bin/bash
export CUDA_VISIBLE_DEVICES=2

# Level 2
python eval.py --model_path /share/nlp/tuwenming/models/m-a-p/MIO-7B-Instruct --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_2/MVIC

# Level 3
python eval.py --model_path /share/nlp/tuwenming/models/m-a-p/MIO-7B-Instruct --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVH

# nohup bash eval2.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_mio_gpu2_$(date +%Y%m%d%H%M%S).log 2>&1 &