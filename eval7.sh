# !/bin/bash
export CUDA_VISIBLE_DEVICES=3

# Level 5
python eval.py --model_path /share/nlp/tuwenming/models/m-a-p/MIO-7B-Instruct --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_5/AVLG
python eval.py --model_path /share/nlp/tuwenming/models/m-a-p/MIO-7B-Instruct --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_5/AVQA

# Level 6
python eval.py --model_path /share/nlp/tuwenming/models/m-a-p/MIO-7B-Instruct --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_6/AVSQA

# nohup bash eval7.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_mio_gpu7_$(date +%Y%m%d%H%M%S).log 2>&1 &