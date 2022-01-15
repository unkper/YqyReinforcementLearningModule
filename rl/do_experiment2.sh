p_num=32
g_size=4
count=3
trainStep=400
maxStep=250
xvfb-run -a python run.py --dir train_10_steady --map map_10 --train_step $trainStep  --max_step $maxStep --threads 2 --p_num=$p_num --g_size=$g_size --count=$count
xvfb-run -a python run.py --dir train_11_steady --map map_11 --train_step $trainStep  --max_step $maxStep --threads 2 --p_num=$p_num --g_size=$g_size --count=$count
maxStep=1000
#xvfb-run -a python run.py --dir train_12_steady --map map_12 --train_step $trainStep  --max_step $maxStep --threads 2 --p_num=$p_num --g_size=$g_size --count=$count
