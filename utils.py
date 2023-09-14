from itertools import product
from collections import Counter
import time
import datetime
import sys
import os

def timeit(func, *args, **kwargs):
    print(f"{(80-len(func.__name__))//2*'-'}{func.__name__}{(80-len(func.__name__))//2*'-'}")
    print("|||||arguments:", args, kwargs, "|||||")
    start_time = time.time()
    result = None
    try:
        result = func(*args, **kwargs)
    finally:
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.6f} seconds to run.")
        print("-"*80)
    return result

def assert_file_exist(filename):
    if not os.path.isfile(filename):
        print(f"Error: File '{filename}' does not exist.")
        sys.exit(1)