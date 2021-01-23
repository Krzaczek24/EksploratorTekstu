import helpers.nltk as nltk
import helpers.files as files
import numpy as np
import re

all_words_file_name = 'all'


def get_lines_by_type(lines, mapping):
    print('Splitting texts by type ... ')
    result = {}
    for map in mapping:
        result[mapping[map]] = []

    for (index, text) in enumerate(lines):
        data = text.split(',')
        try:
            if mapping.get(int(data[1])):
                result[mapping[int(data[1])]].append(data[0])
            else:
                raise Exception(f'Row with index [{index}] has bad value')
        except IndexError:
            raise Exception(f'Row with index [{index}] has no value')

    print('> Texts splitted')
    return result


def get_arrays_subtraction(array1, array2):
    return list(filter(lambda x: x not in array2, array1));


def cosinus_similarity(array1, array2):
    arr1 = np.array(array1)
    arr2 = np.array(array2)
    cos_beta = np.dot(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))
    return cos_beta


def fix_database(input_file_name, output_file_name, lines_count=0, encoding='utf-8', print_times=20):
    input_path = f'{files.files_directory}/{files.sub_folder}/{input_file_name}'
    output_path = f'{files.files_directory}/{files.sub_folder}/{output_file_name}'
    print(f'Source database file path: [{input_path}]')
    print(f'Target database file path: [{output_path}]')
    if lines_count == 0:
        with open(input_path, encoding=encoding, mode='r') as reader:
            line = reader.readline()
            lines_count = -1
            while line:
                line = reader.readline()
                lines_count += 1
    print(f'{lines_count} lines to proceed')
    percent = 0.0
    percents = 100 / print_times
    counter = 0
    global_counter = 0
    step = lines_count / print_times

    print(f'Started fixing database ... ')
    with open(input_path, encoding=encoding, mode='r') as source:
        with open(output_path, encoding=encoding, mode='w') as target:
            print(f'[0.0%] 0 of {lines_count}')
            # header
            line = source.readline().rstrip("\n")
            parts = line.split(',')
            ready_line = parts[0] + ',' + parts[-1] + '\n'
            target.write(ready_line)
            ready_line = ''

            # body
            while True:
                line = source.readline().rstrip("\n")
                if not line:
                    line = ' '
                ready_line += line

                counter += 1
                global_counter += 1

                if counter >= step:
                    counter = 0
                    percent += percents
                    print(f'[{percent}%] {global_counter} of {lines_count}')

                if global_counter > lines_count:
                    print(f'[100.0%] {lines_count} of {lines_count}')
                    print(f'> Fixing database finished successfully')
                    return

                if re.search(".+?,(\d+?\.0)?,-?[01]$", line):
                    prev = ' '
                    while prev != ready_line:
                        prev = ready_line
                        ready_line = ready_line.replace("  ", " ")
                    ready_line = re.sub('^"?(.*?)"?,(\d+?\.0)?,(-?[01])$', r'\1,\3', ready_line) + '\n'
                    parts = ready_line.split(',')
                    ready_line = ' '.join(parts[:-1]) + ',' + parts[-1]
                    target.write(ready_line)
                    ready_line = ''


def load_all_coments(database_file_name):
    return files.load_file_lines(f'{files.sub_folder}/{database_file_name}')[1:]


def lemmatize_and_save_unique_words(comments, stop_words, types, print_times=20, fix_neutral=False, factor=1.5):
    all_unique_words = {}
    unique_words = {}
    for type in types:
        unique_words[type] = {}

    lemmatizer = nltk.get_polish_lemmatizer()

    print('Starting lemmatization and saving unique words ... ')
    for type in unique_words:
        comments_count = len(comments[type])

        print(f'Processing {comments_count} {type.rstrip("_debug")} comments ... ')
        print(f'[0.0%] 0 of {comments_count}')

        percent = 0.0
        percents = 100 / print_times
        counter = 0
        global_counter = 0
        step = comments_count / print_times

        for comment in comments[type]:
            counter += 1
            global_counter += 1

            if counter >= step:
                counter = 0
                percent += percents
                print(f'[{percent}%] {global_counter} of {comments_count}')

            _comment = re.sub(r'[\\\d~`!@#$%^&*()_+{}|:"<>?,./;\'\[\]\-=]', '', comment).lower()
            words = nltk.get_lemmatized_polish_words(_comment, lemmatizer)
            for word in words:
                word_lemma = re.sub('\s', r'', word.lemma_)
                if word_lemma not in stop_words and word_lemma != '':
                    if unique_words[type].get(word_lemma) is None:
                        unique_words[type][word_lemma] = 1
                    else:
                        unique_words[type][word_lemma] += 1

        print(f'[100.0%] {global_counter} of {comments_count}')
        print(f'> Finished processing {type} comments')

        print(f'Adding {type} words to common set ... ')
        for word in unique_words[type]:
            if all_unique_words.get(word) is None:
                all_unique_words[word] = 1
            else:
                all_unique_words[word] += 1
        print(f'> Finished addition of {type} words')

    print('Saving processed words ... ')
    for type in unique_words:
        if type == 'neutral' and fix_neutral:
            prepare_neutral_words(unique_words, all_unique_words, factor)
        save_unique_words(type, unique_words[type])
    save_unique_words(all_words_file_name, all_unique_words)
    print(f'> Saved all words')

    unique_words[all_words_file_name] = all_unique_words

    return unique_words


def save_unique_words(type, words):
    lines = list(map(lambda x: f'{x[0]},{x[1]}', words.items()))
    files.save_to_file(lines, f'{type}.txt')


def load_ready_words(types):
    print('Loading ready words ... ')

    unique_words = {}

    for type in types:
        lines = files.load_file_lines(f'{files.sub_folder}/{type}.txt')
        unique_words[type] = unstringify_dictionary(lines)
    lines = files.load_file_lines(f'{files.sub_folder}/{all_words_file_name}.txt')

    unique_words[all_words_file_name] = unstringify_dictionary(lines)

    print('Loading finished')
    return unique_words


def unstringify_dictionary(lines):
    result = {}
    data = list(map(lambda x: x.rstrip('\n').split(','), lines))
    for entry in data:
        result[entry[0]] = entry[1]
    return result


def generate_debug_database(file_name, percent=1, encoding='utf-8'):
    input_path = f'{files.files_directory}/default/{file_name}'
    output_path = f'{files.files_directory}/debug/{file_name}'

    lines = 0
    counter = 0

    with open(input_path, encoding=encoding, mode='r') as reader:
        line = 'count'
        while line:
            line = reader.readline()
            lines += 1

    step = int(lines * percent / 100)

    with open(input_path, encoding=encoding, mode='r') as source:
        with open(output_path, encoding=encoding, mode='w') as target:
            line = source.readline()
            target.write(line)
            counter += 1
            while line:
                if counter == step:
                    target.write(line)
                    counter = 0
                line = source.readline()
                counter += 1


def prepare_neutral_words(unique_words, all_unique_words, factor):
    for word in all_unique_words:
        if word in unique_words['positive'] and word in unique_words['negative']:
            positive = int(unique_words['positive'][word])
            negative = int(unique_words['negative'][word])
            if max(positive, negative) <= min(positive, negative) * factor:
                neutral = int((positive + negative) / 2)
                unique_words['neutral'][word] = neutral
