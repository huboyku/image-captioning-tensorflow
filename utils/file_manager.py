import os 

FOLDER_LIST = [
    "utils",
    "src",
    "data/images",
    "data/captions",
    "features",
    "models",
    "outputs/predictions"
]

FILE_LIST = [
    "utils/file_manager.py",
    "utils/data_downloader.py",
    "utils/__init__.py",         # makes 'utils' a package (can be empty)
    "src/__init__.py",           # makes 'src' a package (can be empty)
    "src/data_utils.py",         # file reading, parsing, flattening
    "src/text_processing.py",    # cleaning text/captions
    "src/tokenizer_utils.py",    # vectorizer/tokenizer logic
    "src/data_generator.py",     # tf.data.Dataset logic
    "src/feature_extractor.py",  # CNN features
    "src/model.py",              # model code
    "src/train.py",              # training loops
    "src/evaluate.py",           # evaluation
]


def create_folders(folder_list = FOLDER_LIST):
    for folder in folder_list:
        os.makedirs(folder,exist_ok=True)
    
def create_files(file_name,content):
    try:
        with open(file_name,'x') as f:
            f.write(content)
    except FileExistsError:
        print(f"Error: File '{file_name}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")
    