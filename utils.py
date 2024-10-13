import time
from functools import wraps


def timer(func_name=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            name = func_name or func.__name__
            print(f"Operation '{name}' took {elapsed_time:.6f} seconds to run.")
            return result

        return wrapper

    return decorator
