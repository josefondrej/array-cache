import numpy

from array_cache.cache import ArrayCache

TEST_CACHE_NAME = 'test'

if __name__ == '__main__':
    def generate_ten_triples():
        for i in range(10):
            array = numpy.random.randn(3)
            print(array)
            yield array


    cache = ArrayCache(name=TEST_CACHE_NAME, array_generator=generate_ten_triples())
    cache.start_caching_array_generator()
