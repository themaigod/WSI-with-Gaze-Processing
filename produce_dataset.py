import dataProcesser as Da
import openslide
import cv2
import numpy as np
import os
import random
import itertools
from tool.Error import (RegisterError, ModeError, ExistError, OutBoundError)
from tool.manager import (Static, Manager)


# this file is going to produce dataset in different conditions.

class GetDataset:  # 标准父类
    def __init__(self):
        self.order = []  # 类执行过的函数
        self.func_name = []  # 类所有可用函数名
        self.func_setting = []  # 可用函数配对的设置
        self.static_func_name = []  # 静态函数名集合
        self.static_func_setting = []  # 静态函数设置集合
        self.func_manager = Manager(self)
        self.static = Static(self)

    def detect_background(self):
        order = ["background"]
        self.order.append(order)

    def detect_outside(self):
        order = ["outside"]
        self.order.append(order)

    def detect_edge(self):
        order = ["edge"]
        self.order.append(order)

    def level_func(self):
        order = ["level"]
        self.order.append(order)

    def data_func(self):
        order = ["data"]
        self.order.append(order)

    def point_func(self):
        order = ["point"]
        self.order.append(order)

    def repeat_func(self):
        order = ["repeat"]
        self.order.append(order)

    def pixel_func(self):
        order = ["pixel"]
        self.order.append(order)

    def zero_func(self):
        order = ["zero"]
        self.order.append(order)

    def read_image(self):
        order = ["read_image"]
        self.order.append(order)

    def read_image_area(self):
        order = ["read_image_area"]
        self.order.append(order)

    def set_label(self):
        order = ["set_label"]
        self.order.append(order)

    def process(self):
        order = ["process"]
        self.order.append(order)

    def save(self):
        order = ["save"]
        self.order.append(order)

    def get_order(self):
        s = 0
        for i in range(len(self.order)):
            if i < len(self.order) - 2:
                if self.order[i] != self.order[i + 1]:
                    print(self.order[i])
                    if s != 0:
                        print("num:{}".format(s))
                    s = 0
                else:
                    s = s + 1
            else:
                print(self.order[i])
        return self.order

    def __getitem__(self, item):
        order = ["getitem: {}".format(item)]
        self.order.append(order)

    def __len__(self):
        order = ["len"]
        self.order.append(order)

    def __next__(self):
        order = ["next"]
        self.order.append(order)


class GetSampleDataset(GetDataset):
    def __init__(self):  # 读取数据初始化
        super(GetSampleDataset, self).__init__()
        print("starting reader...")
        [name, point_array, level_array, path, max_num, image_label] = Da.allreader(ercpath)
        print("finish read point file")
        print("max num:{}".format(sum(max_num)))
        self.name_array = name
        self.point_array = point_array
        self.level_array = level_array
        self.path = path
        self.max_num = max_num
        self.image_label = image_label

    def name2path(self, name, input_direc):
        """
        :param name:  filepath when erc is saved
        :param input_direc: real data filepath
        :return: all files' path which can be read now
        """
        name_refactor = name.replace("\\", '/')
        input_refactor = input_direc.replace("\\", '/')
        [_, true_name] = os.path.split(name_refactor)
        if input_direc[-1] == '/' or input_direc[-1] == "\\":
            new_path = input_refactor + true_name
        else:
            new_path = input_refactor + "/" + true_name
        self.static_func_register(self.name2path)
        self.static_func_record()
        return new_path

    def name2path_in_list(self, name, input_direc):
        """
        :param name: filepath when erc is saved
        :param input_direc:  real data filepath
        :return:  all files' path which can be read now
        """
        path_list = []
        for i in range(len(name)):
            path = name2path(name[i], input_direc)
            path_list.append(path)
        self.static_func_register(self.name2path_in_list)
        return path_list
