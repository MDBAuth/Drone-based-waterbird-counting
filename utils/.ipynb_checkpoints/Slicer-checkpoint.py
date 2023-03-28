import numpy as np
from itertools import product

class Slicer():
    def __init__(self, w=8192, h=5460, dw=512, dh=546, stride=1):
        self.w = w
        self.h = h
        self.dw = dw
        self.dh = dh
        self.stride = stride
        self.offset = (self.dw*self.stride, self.dh*self.stride)
        self.cols = int(np.ceil(self.w / self.offset[0]))
        self.rows = int(np.ceil(self.h / self.offset[1]))
        
    def get_slice_dict(self):
        slice_dict = {}
        for col, row in product(range(self.cols), range(self.rows)):
            left = int(self.offset[0]*col)
            top = int(self.offset[1]*row)
            right = left + self.dw
            bottom = top + self.dh
            if left>=0 and right<=self.w and top>=0 and bottom<=self.h: 
                slice_dict[(col,row)] = (left, top, right, bottom)
        return slice_dict

    # TODO: Add logic for when stride is included
    def get_slice_id_from_point(self, point):
        x = point['x']
        y = point['y']
        
        id_x1 = int(np.floor(x/self.offset[0]))
        id_x1 = id_x1 if not id_x1 == self.cols else self.cols-1
        id_y1 = int(np.floor(y/self.offset[1]))
        id_y1 = id_y1 if not id_y1 == self.rows else self.rows-1
        return (id_x1, id_y1)