a_star_count=1000
random_count=100
p_num=32
g_size=4
#xvfb-run -a python run.py --map=map_10 --count=$a_star_count --p_num=$p_num --g_size=$g_size
#xvfb-run -a python run.py --map=map_10 --count=$random_count --p_num=$p_num --g_size=$g_size --use_random=True --max_step=2000
#xvfb-run -a python run.py --map=map_11 --count=$a_star_count --p_num=$p_num --g_size=$g_size
#xvfb-run -a python run.py --map=map_11 --count=$random_count --p_num=$p_num --g_size=$g_size --use_random=True --max_step=2000
xvfb-run -a python run.py --map=map_12 --count=$a_star_count --p_num=$p_num --g_size=$g_size --frame_skip=16
xvfb-run -a python run.py --map=map_12 --count=$random_count --p_num=$p_num --g_size=$g_size --use_random=True --max_step=2000 --frame_skip=16
