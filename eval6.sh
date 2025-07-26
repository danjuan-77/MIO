# !/bin/bash
export CUDA_VISIBLE_DEVICES=2

python eval.py --model_path /share/nlp/tuwenming/models/m-a-p/MIO-7B-Instruct --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVLG
python eval.py --model_path /share/nlp/tuwenming/models/m-a-p/MIO-7B-Instruct --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVQA

# nohup bash eval6.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_mio_gpu6_$(date +%Y%m%d%H%M%S).log 2>&1 &