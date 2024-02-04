import csv
import numpy as np


def read_csv_to_dict_list(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        dict_list = [row for row in csv_reader]
    return dict_list


def clean_invalid_line(input_list):
    for line in input_list:
        for key, value in line.items():
            if value == '':
                input_list.remove(line)


def get_batch(X, Y, batch_size):
    assert len(X) == len(Y), "X and Y must have the same length"

    num_samples = len(X)
    num_batches = num_samples // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size

        batch_X = X[start_idx:end_idx]
        batch_Y = Y[start_idx:end_idx]
        yield np.stack(batch_X), np.stack(batch_Y)
