import sys
from bitarray import bitarray
import numpy as np

class BitMap:
    def __init__(self, shape) -> None:
        self.rows = shape[0]
        self.cols = shape[1]
        self.bitmaps = []

    def set_from_bool_array(self, array):
        self.clear_bitmap()
        for val in array:
            if type(val) != list:
                val = val.tolist()

            self.bitmaps.append(bitarray(val))

    def clear_bitmap(self):
        self.bitmaps.clear()

    def at(self, row, col):
        return self.bitmaps[row][col]
    
    def get_size_in_memory(self):
        return sum([sys.getsizeof(b) for b in self.bitmaps]) / 1024 / 1024