p_num=32
g_size=4
count=3
trainStep=400
maxStep=250
#xvfb-run -a python run.py --dir train_10 --map map_10 --train_step $trainStep  --max_step $maxStep --threads 2 --p_num=$p_num --g_size=$g_size --count=$count --use_decay=True
#xvfb-run -a python run.py --dir train_11 --map map_11 --train_step $trainStep  --max_step $maxStep --threads 2 --p_num=$p_num --g_size=$g_size --count=$count --use_decay=True
maxStep=300
xvfb-run -a python run.py --dir train_12 --map map_12 --train_step $trainStep  --max_step $maxStep --threads 2 --p_num=$p_num --g_size=$g_size --count=$count --use_decay=True
