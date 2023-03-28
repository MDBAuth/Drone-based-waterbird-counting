import os
import sys
from pathlib import Path

_true_set = {'yes', 'true', 't', 'y', '1'}
_false_set = {'no', 'false', 'f', 'n', '0'}

def str2bool(value):
    if isinstance(value, str) or sys.version_info[0] < 3 and isinstance(value, basestring):
        value = value.lower()
        if value in _true_set:
            return True
        if value in _false_set:
            return False

def setdiff(list1, list2):
    c = set(list1).union(set(list2))  # or c = set(list1) | set(list2)
    d = set(list1).intersection(set(list2))  # or d = set(list1) & set(list2)
    return list(c - d)

def make_tmp_folder(folder_name, quiet=False):
    try:
        os.makedirs(folder_name)
        return True
    except OSError as e:
        if not quiet:
            print("{} folder already exists".format(folder_name))
        return False

def ensure_path(path: str) -> Path:
    path = Path(path)
    if not os.path.exists(path):
        os.makedirs(path)
    return path