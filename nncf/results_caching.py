import inspect


class ResultsCacheContainer:
    def __init__(self):
        self._cache = {}

    def clear(self):
        self._cache.clear()

    def is_empty(self):
        return len(self._cache) == 0

    def __getitem__(self, item):
        return self._cache[item]

    def __setitem__(self, key, value):
        self._cache[key] = value

    def __contains__(self, item):
        return item in self._cache


def cache_results(cache: ResultsCacheContainer):
    def decorator(func):
        def wrapper(*args, disable_caching=False, **kwargs):
            sig = inspect.signature(func)
            new_kwargs = {name: arg for name, arg in zip(sig.parameters, args)}
            new_kwargs.update(kwargs)
            cache_key = (func.__name__, frozenset(new_kwargs.items()))
            if cache_key in cache:
                return cache[cache_key]
            result = func(*args, **kwargs)
            if not disable_caching:
                cache[cache_key] = result
            return result

        return wrapper

    return decorator
