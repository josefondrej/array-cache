import ast
import os
import shutil
import tempfile
from multiprocessing import shared_memory
from multiprocessing.shared_memory import SharedMemory
from threading import Lock
from typing import Generator, List

import numpy
import pandas as pandas
import sqlalchemy
from pandas import Series
from watchgod import watch

ITEMS_TABLE_NAME = 'items'
GENERATOR_END_INDICATOR_FILE_NAME = 'generator_finished'


class ArrayCache:
    def __init__(self, name: str, array_generator: Generator = None,
                 memory_limit: int = 1000000):
        self._name = name
        self._array_generator = array_generator
        self._memory_limit = memory_limit

    @property
    def name(self):
        return self._name

    def log(self, message: str):
        formatted_message = f'[ArrayCache] {message}'
        print(formatted_message)

    def start_caching_array_generator(self):
        self.log('Starting to cache arrays')
        self.initialize()

        for array in self._array_generator:
            while not self.is_sufficient_memory_available():
                self.log('Insufficient memory, blocking until change')
                self.block_until_cache_dir_change()
            self.add_array_to_shared_memory(array)

        self.mark_generator_end()
        self.block()

    def get_metadata_dir_path(self) -> str:
        metadata_dir = tempfile.gettempdir().replace('\\', '/') + '/' + f'cache_{self.name}/'
        os.makedirs(metadata_dir, exist_ok=True)
        return metadata_dir

    def get_metadata_file_path(self) -> str:
        return self.get_metadata_dir_path() + 'metadata.sqlite'

    def get_metadata_connection_string(self) -> str:
        return 'sqlite:///' + self.get_metadata_file_path()

    def get_metadata_engine(self) -> str:
        metadata_connection_string = self.get_metadata_connection_string()
        engine = sqlalchemy.create_engine(metadata_connection_string)
        return engine

    def initialize(self):
        self.initialize_metadata()
        self._shms = list()

    def initialize_metadata(self):
        self.log('Initializing cached array metadata')
        metadata_dir = self.get_metadata_dir_path()
        shutil.rmtree(metadata_dir, ignore_errors=True)
        engine = self.get_metadata_engine()
        sql = f'create table {ITEMS_TABLE_NAME}(name text, size int, shape text, dtype text)'
        engine.execute(sql)

    def load_metadata(self):
        engine = self.get_metadata_engine()
        metadata = pandas.read_sql(ITEMS_TABLE_NAME, engine)
        return metadata

    def save_metadata(self, metadata: pandas.DataFrame):
        engine = self.get_metadata_engine()
        metadata.to_sql(ITEMS_TABLE_NAME, engine, if_exists='replace', index=False)

    def add_item_to_metadata(self, name: str, size: int, shape: List[int], dtype: str):
        item_metadata = {
            'name': name,
            'size': size,
            'shape': shape,
            'dtype': dtype
        }

        metadata = self.load_metadata()
        metadata = metadata.append(item_metadata, ignore_index=True)
        self.save_metadata(metadata)

    def get_generator_end_indicator_file_path(self) -> str:
        return self.get_metadata_dir_path() + GENERATOR_END_INDICATOR_FILE_NAME

    def mark_generator_end(self):
        self.log('Marking generator end')
        with open(self.get_generator_end_indicator_file_path(), 'w'):
            pass

    def add_array_to_shared_memory(self, array: numpy.ndarray):
        self.log('Adding array to shared memory')
        shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
        self.register_shm(shm=shm)
        shared_array = numpy.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        shared_array[:] = array[:]
        self.add_item_to_metadata(
            name=shm.name,
            size=array.nbytes,
            shape=str(array.shape),
            dtype=str(array.dtype)
        )

    def get_next_array(self) -> numpy.ndarray:
        if self.get_cached_array_count() < 1:
            if self.is_generator_end():
                raise StopIteration
            else:
                while self.get_cached_array_count() < 1:
                    self.block_until_cache_dir_change()

        self.log('Getting next array')
        metadata_item = self.pop_metadata_item()
        self.log('  -- converting metadata to array')
        array = self.array_from_metadata(metadata_item)
        return array

    def pop_metadata_item(self) -> pandas.Series:
        metadata = self.load_metadata()
        item = metadata.iloc[0, :]
        remaining_metadata = metadata.iloc[1:, :]
        self.save_metadata(remaining_metadata)
        return item

    def block_until_cache_dir_change(self):
        for change in watch(path=self.get_metadata_dir_path()):
            break

    def is_sufficient_memory_available(self) -> bool:
        return self.get_total_shared_memory_size() < self._memory_limit

    def get_total_shared_memory_size(self) -> int:
        metadata = self.load_metadata()
        return int(metadata['size'].sum())

    def is_generator_end(self):
        return os.path.exists(self.get_generator_end_indicator_file_path())

    def get_cached_array_count(self) -> int:
        metadata = self.load_metadata()
        return len(metadata)

    def block(self):
        self.log('Blocking')
        lock = Lock()
        lock.acquire()
        lock.acquire()

    def array_from_metadata(self, metadata_item: Series):
        shm = shared_memory.SharedMemory(name=metadata_item['name'])
        array_shape = ast.literal_eval(metadata_item['shape'])
        array = numpy.ndarray(
            shape=array_shape,
            dtype=metadata_item['dtype'],
            buffer=shm.buf
        )
        shm.close()
        shm.unlink()
        return array

    def register_shm(self, shm: SharedMemory):
        self._shms.append(shm)
