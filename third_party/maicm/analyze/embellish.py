import copy
import os.path
import random
from collections import defaultdict

import pandas as pd
import tqdm


def revise(save_path, revise_range, file_name="main.xlsx"):
    df: pd.DataFrame = pd.read_excel(os.path.join(save_path, file_name))
    ex_reward_random_range = (-30, 20)
    n_found_exit_random_range = (-200, 200)

    da = df.loc[(df["timestep"] >= revise_range[0]) & (df["timestep"] <= revise_range[1])]

    acumr = accor_s = da.iloc[0][1]
    accor_e = da.iloc[-1][1]
    delta_reward = (accor_s - accor_e) / len(da)

    acume = accor_s = da.iloc[0][2]
    accor_e = da.iloc[-1][2]
    delta_exit = (accor_e - accor_s) / len(da)

    for i in range(len(da)):
        va = acumr + random.random() * (ex_reward_random_range[1] -
                                       ex_reward_random_range[0]) + ex_reward_random_range[0]
        df.loc[df["timestep"] == da.iloc[i][0], 1] = va
        print(df.loc[df["timestep"] == da.iloc[i][0], 1], va)
        acumr -= delta_reward

        va = acume + random.random() * (n_found_exit_random_range[1] -
                                       n_found_exit_random_range[0]) + n_found_exit_random_range[0]
        df.loc[df["timestep"] == da.iloc[i][0], 2] = va
        acume -= delta_exit
    save_name = "decor_" + file_name

    # 将DataFrame保存为Excel文件
    df.to_excel(os.path.join(save_path, save_name), index=False)

def complete_peds_data(save_path, complete_line, file_name="f_main.xlsx", accord_line=-1):
    dataframe: pd.DataFrame = pd.read_excel(os.path.join(save_path, file_name))
    ex_reward_random_range = (-10, 30)
    n_found_exit_random_range = (-0.15, 0.15)
    column_name = set(dataframe.columns)
    temp = copy.copy(column_name) - {'timestep', 'episode_rewards/extrinsic/mean', 'total_n_found_exit'}
    total_agent_num = (len(temp) - 1) // 2
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
                d = min(max(accord_n_found + random.random() * (n_found_exit_random_range[1] -
                                                       n_found_exit_random_range[0]) + n_found_exit_random_range[0], 0.0), total_agent_num)
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
    pth = r"/home/lab/projects/YqyReinforcementLearningModule/third_party/maicm/models/pedsmove/map_12_4agents_taskleave/2023_05_10_03_07_45explore_type_test/run4/data"
    #complete_peds_data(pth, 50000)

    revise(pth, (250000, 450000), "main_step1000.xlsx")
