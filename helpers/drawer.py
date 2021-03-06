import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# pip install wordcloud
# conda install -c conda-forge wordcloud


def draw_tokenized_words_chart(tokenized_words, top=10):
    tokenized_words.plot(top, cumulative=False)
    plt.show()


def draw_word_cloud(type, type_words_freq, font='HussarNiebieski-1rm4.ttf'):
    # [(word, amount), (word, amount), ...]

    strings = []
    for word in type_words_freq:
        for index in range(type_words_freq[word]):
            strings.append(word)

    width = 10.24  # x * 100 = szerokość całego okna w px
    height = 7.68  # y * 100 = wysokość całego okna w px

    plt.figure(figsize=(width, height))

    width = int(width * 100)
    height = int(height * 100)

    wordcloud = WordCloud(font_path=f'files/{font}', width=width, height=height, collocations=False)
    wordcloud.generate(' '.join(strings))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title(type)
    plt.show()


def draw_bar_chart(title, data):
    columns = [column for column in data]
    values = [data[column] for column in data]
    y_pos = np.arange(len(columns))

    plt.bar(y_pos, values, align='center', alpha=0.5)
    plt.xticks(y_pos, columns)
    plt.ylabel('amount')
    plt.title(f'{title} words')
    plt.show()


def draw_table(title, data, root_cell_text=''):
    headers = [root_cell_text] + [row[0] for row in data]

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)  # ukrycie linii
    fig.set_size_inches(10, 6)
    ax.axis('off')  # ukrycie osi
    df = pd.DataFrame(data, columns=headers)
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    table.scale(1, 3)
    table.set_fontsize(24)
    plt.title(title, fontsize=32)
    plt.show()
