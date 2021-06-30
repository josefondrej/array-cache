from array_cache.cache import ArrayCache
from array_cache.cache_producer import TEST_CACHE_NAME

if __name__ == '__main__':
    cache = ArrayCache(name=TEST_CACHE_NAME)
    while True:
        try:
            next_array = cache.get_next_array()
            print(f'Retrieved array: {str(next_array)}')
        except StopIteration:
            break
