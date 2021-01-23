import io
import pandas as pd
import numpy as np
from os import listdir

files_directory = 'files'
sub_folder = 'standard'


def does_file_exists(file_name):
    file_path = f'{files_directory}/{sub_folder}/{file_name}'
    try:
        open(file_path).close()
        return True
    except IOError:
        return False


def does_all_type_files_exists(types):
    all_words_file_name = 'all'
    all_exists = np.all([does_file_exists(f'{type}.txt') for type in types])
    all_exists = all_exists and does_file_exists(f'{all_words_file_name}.txt')
    return all_exists


def get_files(path=''):
    if path == '':
        return listdir(files_directory)
    return listdir(path)


def load_csv_file(file_name, delimiter=','):
    file_path = f'{files_directory}/{sub_folder}/{file_name}'
    return pd.read_csv(file_path, delimiter=delimiter)


def load_file_text(file_name, encoding='utf-8'):
    file_path = f'{files_directory}/{sub_folder}/{file_name}'
    reader = io.open(file_path, mode='r', encoding=encoding)
    text = reader.read()
    reader.close()
    return text


def load_file_lines(file_name, encoding='utf-8', new_line='\n'):
    file_path = f'{files_directory}/{file_name}'
    lines = []

    with open(file_path, encoding=encoding, mode='r') as source:
        line = source.readline().rstrip(new_line)
        while line:
            lines.append(line)
            line = source.readline().rstrip(new_line)

    if len(lines) <= 0:
        return []

    while lines[-1] == '':
        lines.pop()
        if len(lines) <= 0:
            return []

    return lines


def save_to_file(text_lines, file_name, encoding='utf-8'):
    with open(f'{files_directory}/{sub_folder}/{file_name}', encoding=encoding, mode='w') as target:
        for line in text_lines:
            target.write(line + '\n')
