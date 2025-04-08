import random
from datetime import datetime
import os


def get_randomID(
    length:int=4
):
    random_chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return ''.join(random.choices(random_chars, k=length))

def create_dir(
    prefix:str, 
    date:bool=True, 
    randomID:bool=True
):
    if date:
        prefix += f"_{datetime.now().strftime('%Y%m%d')}"
    if randomID:
        prefix += f"_{get_randomID(4)}"
    os.makedirs(prefix, exist_ok=True)
    return prefix
