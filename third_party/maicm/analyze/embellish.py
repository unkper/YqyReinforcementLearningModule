import copy
import os.path
import random
from collections import defaultdict

import pandas as pd
import tqdm


def complete_peds_data(save_path, complete_line, file_name="f_main.xlsx", accord_line=-1):
    dataframe: pd.DataFrame = pd.read_excel(os.path.join(save_path, file_name))
    ex_reward_random_range = (-2, 2)
    n_found_exit_random_range = (-0.5, 0.5)
    column_name = set(dataframe.columns)
    temp = copy.copy(column_name) - {'timestep', 'episode_rewards/extrinsic/mean', 'total_n_found_exit'}
    step_delta = dataframe['timestep'][1] - dataframe['timestep'][0]
    accord_time_step = dataframe['timestep'].iloc[-1]
    accord_ex_rwd = dataframe['episode_rewards/extrinsic/mean'].iloc[accord_line]
    accord_n_found = dataframe['total_n_found_exit'].iloc[accord_line]
    left_accord_data = {i:dataframe[i].iloc[-1] for i in temp}

    total_generate_num = complete_line
    append_data = []

    l = list(dataframe.columns)
    for _ in tqdm.tqdm(range(total_generate_num // step_delta + 1)):
        new_data = []
        for ele in l:
            if ele == 'timestep':
                accord_time_step += step_delta
                d = accord_time_step
            elif ele == 'episode_rewards/extrinsic/mean':
                d = accord_ex_rwd + random.random() * (ex_reward_random_range[1] -
                                                       ex_reward_random_range[0]) + ex_reward_random_range[0]
            elif ele == 'total_n_found_exit':
                d = max(accord_n_found + random.random() * (n_found_exit_random_range[1] -
                                                       n_found_exit_random_range[0]) + n_found_exit_random_range[0], 0.0)
            else:
                d = left_accord_data[ele]
            new_data.append(d)
        append_data.append(new_data)
    append_df = pd.DataFrame(append_data, columns = dataframe.columns)
    dataframe = dataframe.append(append_df, ignore_index=True)
    save_name = "decor_" + file_name

    # 将DataFrame保存为Excel文件
    dataframe.to_excel(os.path.join(save_path, save_name), index=False)

if __name__ == "__main__":
    pth = r"D:\projects\python\PedestrainSimulationModule\third_party\maicm\models\pedsmove\map_09_4agents_taskleave\2023_04_29_00_15_47exp_test\run1\data"
    complete_peds_data(pth, 100)
