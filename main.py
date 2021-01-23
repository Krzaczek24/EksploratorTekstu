# Moje nakładki / narzędzia
import helpers.files as files
import helpers.tools as tools

DEBUG = False
FORCE_GEN_DEBUG_DB = False
FORCE_FIX_DATABASE = False
FORCE_SAVE_WORDS = False
FIX_NEUTRAL = True
SHOW_WORD_CLOUD = False
SHOW_EMOTION_EVAL = False
SHOW_COSINUS_SIMILARITY = False
database_to_fix_name = 'polish_sentiment_dataset'
database_name = 'fixed_database'
db_extension = 'csv'
print_times = 20
mapping = {1: 'positive', 0: 'neutral', -1: 'negative'}
emotion_names = {
    'H': 'Happiness',
    'A': 'Anger',
    'S': 'Sadness',
    'F': 'Fear',
    'D': 'Disgust',
    'N': 'Neutral',
    'U': 'Unclassified'
}
types = [mapping[map] for map in mapping]
unique_words = {}

if DEBUG:
    files.sub_folder = 'debug'

    if FORCE_GEN_DEBUG_DB or not files.does_file_exists(f'{database_to_fix_name}.{db_extension}'):
        tools.generate_debug_database(f'{database_to_fix_name}.{db_extension}', 0.1)


if FORCE_FIX_DATABASE or not files.does_file_exists(f'{database_name}.{db_extension}'):
    tools.fix_database(f'{database_to_fix_name}.{db_extension}',
                       f'{database_name}.{db_extension}', print_times=print_times)

if FORCE_SAVE_WORDS or not files.does_all_type_files_exists(types):
    print('Loading raw data from database ... ')
    all_comments = tools.load_all_coments(f'{database_name}.{db_extension}')
    print('> Raw data loaded')
    comments = tools.get_comments_by_type(all_comments, mapping)
    # stop_words = tools.get_custom_polish_stopwords()
    unique_words = tools.lemmatize_and_save_unique_words(comments, types,  # stop_words,
                                                         print_times=print_times, fix_neutral=FIX_NEUTRAL, factor=2.0)
else:
    unique_words = tools.load_ready_words(types)

if SHOW_WORD_CLOUD:
    top_words = tools.normalize_words_amount(unique_words, top_elements=200, target_max_value=20)
    tools.draw_word_clouds(top_words)

if SHOW_EMOTION_EVAL or SHOW_COSINUS_SIMILARITY:
    emotion_definitions = tools.load_word_emotions()
    word_type_emotions, word_emotion_types = tools.convert_words_to_emotions(unique_words,
                                                                             emotion_definitions,
                                                                             emotion_names)
    if SHOW_EMOTION_EVAL:
        tools.draw_types_and_emotions_charts(word_type_emotions, word_emotion_types)

    if SHOW_COSINUS_SIMILARITY:
        tools.draw_cosinus_similarity_table('Cosinus similarity for type emotions', word_type_emotions)
        tools.draw_cosinus_similarity_table('Cosinus similarity for emotion types', word_emotion_types)
