import time


def measure_runtime(func):
    def wrapper(*args, **kwargs):
        n = 10
        start_time = time.time()
        result = [func(*args, **kwargs) for _ in range(n)]
        end_time = time.time()
        runtime = round((end_time - start_time) / n, 5)
        print(f"Runtime of {func.__name__}: {runtime} seconds")
        return result

    return wrapper
