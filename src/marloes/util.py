import time


def timethis(func):
    """
    Decorator to measure the time a function takes to run.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        if args and hasattr(args[0], "__class__"):
            class_name = args[0].__class__.__name__
            print(
                f"Function {class_name}.{func.__name__} took {end_time - start_time:.6f} seconds"
            )
        else:
            print(f"Function {func.__name__} took {end_time - start_time:.6f} seconds")

        return result

    return wrapper
