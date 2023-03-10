p_num=32
g_size=4
count=1
trainStep=50
maxStep=250
xvfb-run -a python run.py --dir debug_11 --map map_11 --train_step $trainStep  --max_step $maxStep --threads 2 --p_num=$p_num --g_size=$g_size --count=$count
