import json
import logging
import os
import sys

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def result_to_text_file(result: dict, file_name: str, verbose: bool = True) -> None:
    with open(file_name, "a") as writer:
        for key in sorted(result.keys()):
            writer.write("%s = %s\n" % (key, str(result[key])))
        writer.write("")


def dictionary_to_json(dictionary: dict, file_name: str):
    with open(file_name, "w") as f:
        json.dump(dictionary, f, indent=2)


def is_folder_empty(folder_name: str):
    if len([f for f in os.listdir(folder_name) if not f.startswith('.')]) == 0:
        return True
    else:
        return False


def get_immediate_subdirectories(directory: str):
    return [os.path.join(directory, name) for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]