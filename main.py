# Moje nakładki / narzędzia
import helpers.files as files
import helpers.tools as tools
import helpers.nltk as nltk

DEBUG = True
FORCE_GEN_DEBUG_DB = False
FORCE_FIX_DATABASE = True
FORCE_SAVE_WORDS = True
FIX_NEUTRAL = True
database_to_fix_name = 'polish_sentiment_dataset'
database_name = 'fixed_database'
db_extension = 'csv'
print_times = 20
mapping = {
    1: 'positive',
    0: 'neutral',
    -1: 'negative'
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
    comments = tools.get_lines_by_type(all_comments, mapping)
    stop_words = nltk.get_custom_polish_stopwords()
    unique_words = tools.lemmatize_and_save_unique_words(comments, stop_words, types,
                                                         print_times, FIX_NEUTRAL, factor=2.0)
else:
    unique_words = tools.load_ready_words(types)

i = 0
