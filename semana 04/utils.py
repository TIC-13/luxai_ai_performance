
import os
from datetime import datetime


def logging(message):
    print(f'{datetime.now().strftime("%Y-%M-%d %H:%M:%S")} {message}')


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False


def get_current_time(format="%Y%m%d_%H%M%S"):
    return datetime.now().strftime(format)