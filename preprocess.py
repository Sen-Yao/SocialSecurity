import csv
import numpy as np
import random


def read_csv_to_dict_list(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        dict_list = [row for row in csv_reader]
        rand_list = dict_list.copy()
        random.shuffle(rand_list)
    return rand_list


def clean_invalid_line(input_list):
    for line in input_list:
        for key, value in line.items():
            if value == '':
                input_list.remove(line)
    for line in input_list:
        for key, value in line.items():
            if value == '':
                input_list.remove(line)


def get_batch(X, Y, batch_size):
    assert len(X) == len(Y), "X and Y must have the same length"
    while True:
        yield np.stack(random.sample(X, batch_size)), np.stack(random.sample(Y, batch_size))

    num_samples = len(X)
    num_batches = num_samples // batch_size

    while True:
        batch_idx = random.randint(0, num_batches-1)
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size

        batch_X = X[start_idx:end_idx]
        batch_Y = Y[start_idx:end_idx]
        yield np.stack(batch_X), np.stack(batch_Y)


