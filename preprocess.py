import csv
import numpy as np
import random
from torch.utils import data


def read_csv_to_dict_list(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        dict_list = [row for row in csv_reader]
        rand_list = dict_list.copy()
        random.shuffle(rand_list)
        city_code_index = get_city_codes_index(rand_list)
    return rand_list, city_code_index


def clean_invalid_line(input_list):
    for line in input_list:
        for key, value in line.items():
            if value == '':
                input_list.remove(line)
    for line in input_list:
        for key, value in line.items():
            if value == '':
                input_list.remove(line)


def get_city_codes_index(input_list):
    city_codes = [int(line['citycode2']) for line in input_list]
    unique_city_codes = list(set(city_codes))
    city_code_to_index = {code: i for i, code in enumerate(unique_city_codes)}
    return city_code_to_index


def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
