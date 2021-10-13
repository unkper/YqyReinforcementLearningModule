from typing import List

import numpy as np
import torch
import socialforce
import copy

from ped_env.utils.maps import Map


def create_one_wall(x, y, horizon: bool, length=1, delta=0.1) -> torch.Tensor:
    """
    在以x,y为原点初创建一堵1m的墙，返回创建好的点位数组
    :param delta:
    :param length:
    :param horizon:
    :param x:
    :param y:
    :return:
    """
    if horizon:
        return torch.stack(
            [torch.linspace(x, x + length, int(length // delta)), torch.full((int(length // delta),), y)], -1)
    else:
        return torch.stack(
            [torch.full((int(length // delta),), x), torch.linspace(y, y + length, int(length // delta))], -1)


def create_square(x, y, length=1) -> List[torch.Tensor]:
    """
    在以x,y为左下角点处创建一个空心矩形，返回组成矩形的列表
    :param x:
    :param y:
    :param length:
    :return:
    """
    under_wall = create_one_wall(x, y, True, length)
    left_wall = create_one_wall(x, y, False, length)
    right_wall = create_one_wall(x + length, y, False, length)
    up_wall = create_one_wall(x, y + length, True, length)
    return [under_wall, left_wall, right_wall, up_wall]

def desired_directions(state):
    """Given the current state and destination, compute desired direction."""
    destination_vectors = state[:, 4:6] - state[:, 0:2]
    norm_factors = torch.linalg.vector_norm(destination_vectors, axis=-1)
    return destination_vectors / torch.unsqueeze(norm_factors, -1)

def parse_map_for_space(terrain: Map):
    """
    根据map返回设置好墙壁的PedSpace对象和map，针对值为1的构造一个空心的正方形障碍
    :param terrain: 要构造的地图对象
    :return:
    """
    barriers = []
    maps = terrain.map
    for j in range(maps.shape[1]):
        for i in range(maps.shape[0]):
            if maps[i, j] == 1 or maps[i, j] == 2:
                barriers.extend(create_square(i, j))
    space = socialforce.potentials.PedSpacePotential(barriers)
    return space


def parse_map_for_people(terrain: Map, person_num: int, desired_speed: float = 2.0, seed: int = 42) -> List[torch.Tensor]:
    """
    根据map返回设置好的行人和map,随机在指定的位置构造若干人
    :param desired_speed:
    :param person_num:
    :param seed:
    :param terrain: 要构造的地图对象
    :return:
    """
    _ = torch.manual_seed(seed)

    state = torch.zeros((person_num, 6))
    num_in_div = int(person_num / len(terrain.start_points))  # 在每个区块平均生成人
    # positions
    for i in range(len(terrain.start_points)):
        spawn_points = torch.tensor([terrain.start_points[i] for _ in range(num_in_div)])
        state[i * num_in_div:i * num_in_div + num_in_div, 0:2] = spawn_points + (
                (torch.rand(num_in_div, 2)) - 0.5) * terrain.create_radius
    # destination
    for i in range(len(terrain.exits)):
        exit_points = torch.tensor([terrain.exits[i] for _ in range(num_in_div)]) #生成一个堆叠数组
        state[i * num_in_div:i * num_in_div + num_in_div, 4:6] = exit_points
    # velocity set to desired_speed to destination
    direction = desired_directions(state)
    state[:, 2:4] = direction * desired_speed

    return state
