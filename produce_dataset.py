import dataProcesser as Da
import openslide
import cv2
import numpy as np
import os
import random
import itertools
from tool.Error import (RegisterError, ModeError, ExistError, OutBoundError)
from tool.manager import (Static, Manager, Inner, SingleManager)
from tool.dataProcesser import all_reader


# this file is going to produce dataset in different conditions.

class GetDataset:  # 标准父类
    def __init__(self, record=True, func_mode=0):
        self.func_mode = func_mode  # 使用函数模式，0为使用inner函数，1位使用static函数
        self.record_status = record  # 是否使用record记录函数执行情况
        self.order = []  # 类执行过的函数
        self.func_name = []  # 类所有可用函数名
        self.func_setting = []  # 可用函数配对的设置
        self.static_func_name = []  # 静态函数名集合
        self.static_func_setting = []  # 静态函数设置集合
        self.inner_func_name = []  # 静态函数名集合
        self.inner_func_setting = []  # 静态函数设置集合
        self.func_manager = Manager(self)  # 函数管理器
        self.static = Static(self)  # 静态函数管理器
        self.inner = Inner(self)  # 内部数据操作的函数的管理器
        self.inner_register()  # 类内函数注册
        self.level_manager = SingleManager(self, "level")  # level操作相关函数管理
        self.data_manager = SingleManager(self, "data")  # data操作相关函数管理
        self.point_manager = SingleManager(self, "point")  # point操作相关函数管理
        self.repeat_manager = SingleManager(self, "repeat")  # 重复操作相关函数管理
        self.pixel_manager = SingleManager(self, "pixel")  # pixel操作相关函数管理
        self.zero_manager = SingleManager(self, "zero")  # 获取标签为0的数据相关函数管理

    def inner_register(self):
        self.inner.register(self.__init__)
        self.inner.register(self.inner_register)
        self.inner.register(self.record)
        self.inner.register(self.func_run)
        self.inner.register(self.inner_func_run)
        self.inner.register(self.func_show)
        self.inner.register(self.func_del)
        self.func_manager.register(self.detect_background)
        self.func_manager.register(self.detect_outside)
        self.func_manager.register(self.detect_edge)
        self.func_manager.register(self.level_func)
        self.func_manager.register(self.data_func)
        self.func_manager.register(self.point_func)
        self.func_manager.register(self.repeat_func)
        self.func_manager.register(self.pixel_func)
        self.func_manager.register(self.zero_func)
        self.func_manager.register(self.read_image)
        self.func_manager.register(self.read_image_area)
        self.func_manager.register(self.set_label)
        self.func_manager.register(self.process)
        self.func_manager.register(self.save)
        self.inner.register(self.get_order)
        self.func_manager.register(self.__getitem__)
        self.func_manager.register(self.__len__)
        self.func_manager.register(self.__next__)
        self.record("inner_register", self.record_status)

    def register(self, func_name, func_type=0, input_mode=0):  # 函数注册
        if func_name not in self.func_name:
            self.func_name.append(func_name)
            self.func_setting.append([func_type, input_mode])
        self.record("register", self.record_status)

    def record(self, key, mode=0, record=True):  # 函数记录
        if record is True:
            if mode == 0:
                order = [key]
                self.order.append(order)
            elif mode == 1:
                order = key
                self.order.append(order)
            else:
                raise ModeError("record" + str(mode))

    def func_run(self, func_name, inputs, func_type=0, input_mode=0):  # 在类内运行函数
        if func_name not in self.func_name:
            raise RegisterError(func_name)
        if func_type == 0:
            pass
        elif func_type == 1:
            func_name = eval(func_name)
        else:
            raise ModeError("func_type {}".format(func_type))
        if input_mode == 0:
            result = func_name(*inputs)
            self.record(str(func_name))
        elif input_mode == 1:
            result = func_name(**inputs)
            self.record(str(func_name))
        else:
            raise ModeError("input_mode {}".format(input_mode))
        self.record("func_run", self.record_status)
        return result

    def inner_func_run(self, func_point, inputs, access_mode=0):  # 使用已被注册的函数在类内运行，并使用注册时的运行设置
        if access_mode == 0:
            index = self.func_name.index(func_point)
            func_name = func_point
            func_type = self.func_setting[index][0]
            input_mode = self.func_setting[index][1]
        elif access_mode == 1:
            index = func_point
            func_name = self.func_name[index]
            func_type = self.func_setting[index][0]
            input_mode = self.func_setting[index][1]
        else:
            raise ModeError("access_mode {}".format(access_mode))
        result = self.func_run(func_name, inputs, func_type, input_mode)
        self.record("inner_func_run", self.record_status)
        return result

    def func_show(self):  # 展示已注册函数
        for i in range(len(self.func_name)):
            print(self.func_name[i])
        self.record("func_show", self.record_status)

    def func_del(self, func_point, access_mode):  # 销毁已注册函数
        if access_mode == 0:
            if func_point not in self.func_name:
                raise ExistError("func_name: " + str(func_point))
            index = self.func_name.index(func_point)
            self.func_name.pop(index)
            self.func_setting.pop(index)
        elif access_mode == 1:
            if func_point >= len(self.func_name):
                raise OutBoundError("func_index: " + str(func_point))
            index_all = func_point
            self.func_name.pop(index_all)
            self.func_setting.pop(index_all)
        else:
            raise ModeError("access_mode {}".format(access_mode))
        self.record("func_del", self.record_status)

    def detect_background(self):  # 检测背景，使用请重写
        self.record("detect_background", self.record_status)

    def detect_outside(self):  # 检测是否需要被舍弃，使用请重写
        self.record("detect_outside", self.record_status)

    def detect_edge(self):  # 检测是否在边缘
        self.record("detect_edge", self.record_status)

    def level_func(self):  # 返回level相关函数管理器
        self.record("level_func", self.record_status)
        return self.level_manager

    def data_func(self):  # 返回data相关函数管理器
        self.record("data_func", self.record_status)
        return self.data_manager

    def point_func(self):  # 返回point相关函数管理器
        self.record("point_func", self.record_status)
        return self.point_manager

    def repeat_func(self):  # 返回重复操作相关函数管理器
        self.record("repeat_func", self.record_status)
        return self.repeat_manager

    def pixel_func(self):  # 返回pixel相关函数管理器
        self.record("pixel_func", self.record_status)
        return self.pixel_manager

    def zero_func(self):  # 返回与标签0相关函数管理器
        self.record("zero_func", self.record_status)
        return self.zero_manager

    def read_image(self):  # 读取图像
        self.record("read_image", self.record_status)

    def read_image_area(self):
        self.record("read_image_area", self.record_status)

    def set_label(self):
        self.record("set_label", self.record_status)

    def process(self):
        self.record("process", self.record_status)

    def save(self):
        self.record("save", self.record_status)

    def get_order(self, mode=0):
        if mode == 0:
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
        self.record("get_order", self.record_status)
        return self.order

    def __getitem__(self, item):
        order = ["getitem: {}".format(item)]
        self.record(order, mode=1, record=self.record_status)

    def __len__(self):
        self.record("__len__", self.record_status)

    def __next__(self):
        self.record("__next__", self.record_status)


class GetInitDataset(GetDataset):
    def __init__(self, erc_path, record=True, func_mode=0, init_status=False, repeat_add_mode=True):  # 读取数据初始化
        super().__init__(record, func_mode)
        self.init_status = init_status
        self.repeat_add_mode = repeat_add_mode
        if self.init_status is True:
            print("starting reader...")
            [name, point_array, level_array, path, max_num, image_label] = Da.allreader(erc_path)  # 获取数据
            print("finish read point file")
            print("max num:{}".format(sum(max_num)))
            self.name_array = name
            self.point_array = point_array
            self.level_array = level_array
            self.path = path
            self.max_num = max_num
            self.image_label = image_label
            if self.func_mode == 0:
                self.inner_repeat_add(self.repeat_add_mode)
                self.inner_re_level()
            elif self.func_mode == 1:
                self.static_repeat_add(self.name_array, self.point_array, self.level_array, self.path, self.image_label,
                                       self.repeat_add_mode)
                self.static_re_level(self.level_array)
            else:
                ModeError("func_mode" + str(self.func_mode))

    def inner_register(self):
        super().inner_register()
        self.static.register(self.name2path)
        self.static.register(self.name2path_in_list)
        self.static.register(self.static_repeat_add)
        self.inner.register(self.inner_repeat_add)
        self.level_manager.register(self.re_level_single)
        self.static.register(self.re_level_single)
        self.level_manager.register(self.static_re_level)
        self.static.register(self.static_re_level)
        self.level_manager.register(self.inner_re_level)
        self.inner.register(self.inner_re_level)

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
        self.static.record("self.name2path", self.record_status)
        return new_path

    def name2path_in_list(self, name, input_direc):
        """
        :param name: filepath when erc is saved
        :param input_direc:  real data filepath
        :return:  all files' path which can be read now
        """
        path_list = []
        for i in range(len(name)):
            path = self.name2path(name[i], input_direc)
            path_list.append(path)
        self.static.record("self.name2path", self.record_status)
        return path_list

    def static_repeat_add(self, name_list, point_list, level_list, path_list, image_label,
                          mode=1):  # 对同一张切片重复观看的进行相加,static版本
        if mode == 1:
            check_list = []
            location_list = []
            u = 0
            for o in range(len(name_list)):
                o = o - u
                if path_list[o] not in check_list:
                    check_list.append(path_list[o])
                    location_list.append([path_list[o], o])
                else:
                    index = check_list.index(path_list[o])
                    location = location_list[index][1]
                    point_list[location] = point_list[location] + point_list[o]
                    level_list[location] = level_list[location] + level_list[o]
                    name_list.pop(o)
                    point_list.pop(o)
                    level_list.pop(o)
                    path_list.pop(o)
                    image_label.pop(o)
                    u = u + 1
        elif mode == 0:
            pass
        else:
            raise ModeError("static_repeat_add" + str(mode))

        self.static.record("static_repeat_add", self.record_status)
        return name_list, point_list, level_list, path_list, image_label

    def inner_repeat_add(self, mode=1):  # 对同一张切片重复观看的进行相加，直接对储存数据修改的版本
        if mode == 1:
            check_list = []
            location_list = []
            u = 0
            for o in range(len(self.name_array)):
                o = o - u
                if self.path[o] not in check_list:
                    check_list.append(self.path[o])
                    location_list.append([self.path[o], o])
                else:
                    index = check_list.index(self.path[o])
                    location = location_list[index][1]
                    self.point_array[location] = self.point_array[location] + self.point_array[o]
                    self.level_array[location] = self.level_array[location] + self.level_array[o]
                    self.name_array.pop(o)
                    self.point_array.pop(o)
                    self.level_array.pop(o)
                    self.path.pop(o)
                    self.image_label.pop(o)
                    u = u + 1
        elif mode == 0:
            pass
        else:
            raise ModeError("inner_repeat_add" + str(mode))
        self.static.record("inner_repeat_add", self.record_status)

    def re_level_single(self, level):
        times = 1
        open_slide_level = 0
        while True:
            if level == times:
                level = open_slide_level
                break
            open_slide_level = open_slide_level + 1
            times = times * 2
        self.level_manager.record("re_level_single", self.record_status)
        return level

    def static_re_level(self, level_array):
        for v in range(len(level_array)):
            for w in range(len(level_array[v])):
                self.re_level_single(level_array[v][w])
        self.level_manager.record("static_re_level", self.record_status)
        return level_array

    def inner_re_level(self):
        for v in range(len(self.level_array)):
            for w in range(len(self.level_array[v])):
                self.re_level_single(self.level_array[v][w])
        self.level_manager.record("inner_re_level", self.record_status)


class DatasetRegularProcess(GetDataset):
    def inner_register(self):
        super().inner_register()
        self.static.register(self.static_transform_to_1)
        self.static.register(self.static_get_same_num)
        self.static.register(self.static_calculate_class_num)

    def static_transform_to_1(self, array):
        result = []
        for i in range(len(array)):
            result.append(array[r] / sum(array))
        self.static.record("transform_to_1", self.record_status)
        return result

    def static_get_same_num(self, array):
        same_location = []
        same_value = []
        array = np.array(array)
        array_no_repeat = set(array)
        for i in range(len(array_no_repeat)):
            location = np.argwhere(array == array_no_repeat[i])
            if len(location) != 1:
                same_value.append(array_no_repeat[i])
                same_location.append(location)
        self.static.record("get_same_num", self.record_status)
        return same_value, same_location

    def static_calculate_class_num(self, array: list, total_num: int):
        array_num = [-1 for _ in range(len(array))]
        same_value, same_location = self.get_same_num(array)
        ratio = self.transform_to_1(array)
        count_class = 0
        count_num = 0
        class_num = len(array)
        for i in range(len(same_location)):
            if count_class + len(same_location[i]) < class_num:
                for j in range(len(same_location[i])):
                    select_num = int(ratio[same_location[i][0]] * total_num)
                    array_num[same_location[i][j]] = select_num
                    count_num += select_num
                    count_class += 1
            else:
                for j in range(len(same_location[i])):
                    select_num = int((total_num - count_num) // len(same_location[i]))
                    array_num[same_location[i][j]] = select_num
                    count_num += select_num
                    count_class += 1
        if count_num != total_num:
            unfinished_num = array_num.count(-1)
            for k in range(unfinished_num):
                if k + 1 != unfinished_num:
                    loc = array_num.index(-1)
                    select_num = int(ratio[loc] * total_num)
                    array_num[loc] = select_num
                    count_num += select_num
                    count_class += 1
                else:
                    loc = array_num.index(-1)
                    select_num = int(total_num - count_num)
                    array_num[loc] = select_num
                    count_num += select_num
                    count_class += 1
        self.static.record("static_calculate_class_num", self.record_status)
        return array_num

    def static_random_index(self, total_num: int):
        random_list = list(range(total_num))
        random.shuffle(random_list)
        self.static.record("static_random_index", self.record_status)
        return random_list

    def static_random_class_index(self, array: list, total_num: int):
        use_list = []
        array_num = self.static_calculate_class_num(array, total_num)
        random_list = self.static_random_index(total_num)
        index = 0
        for i in range(len(array_num)):
            class_list = random_list[index: (index + array_num[i])]
            index = index + array_num[i]
            use_list.append(class_list)
        self.static.record("static_random_class_index", self.record_status)
        return use_list

    def read_image(self, path=None, level=3):
        if path is None:
            raise ExistError("path")
        slide = openslide.OpenSlide(path)
        [length_of_img, height_of_img] = slide.level_dimensions[level]
        img = np.array(slide.read_region([0, 0], level, [length_of_img, height_of_img]))[:, :, :3]
        self.static.record("read_image", self.record_status)
        return img, level

    def static_resize_patch(self, img=None, patch_size=None):
        if img is None:
            raise ExistError("img")
        if patch_size is None:
            raise ExistError("patch_size")
        [x, y, _] = np.shape(img)
        w = int(x // patch_size)
        h = int(y // patch_size)
        img = img[0:(w * patch_size), 0:(h * patch_size), :]
        img = cv2.resize(img, (h, w), interpolation=cv2.INTER_AREA)
        self.static.record("static_resize_patch", self.record_status)
        return img

    def static_bgr2gray(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.static.record("static_bgr2gray", self.record_status)
        return img

    def static_thresh(self, img=None):
        ret2, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.static.record("static_thresh", self.record_status)
        return img

    def static_thresh_patch(self, img, patch_size):
        img = self.static_resize_patch(img, patch_size)
        img = self.static_bgr2gray(img)
        img = self.static_thresh(img)
        self.static.record("static_thresh_patch", self.record_status)
        return img

    def static_read_slide(self, path):
        slide = openslide.OpenSlide(path)
        level_downsamples = slide.level_downsamples
        self.static.record("static_thresh_patch", self.record_status)
        return slide, level_downsamples

    def static_mul_div(self, a_list, mul=None, div=None, as_int=False, to_np=False):
        a_list = np.array(a_list)
        if mul is not None:
            a_list = a_list * mul
        if div is not None:
            a_list = a_list / div
        if as_int is True:
            a_list = a_list.astype(int)
        if to_np is False:
            a_list = a_list.tolist()
        self.static.record("mul_div", self.record_status)
        return a_list

    def static_double_mul_div_int_mul_level(self, x, level_downsamples, level, point_pixel=0):
        x = self.static_mul_div(x, mul=level_downsamples[point_pixel], div=level_downsamples[level], as_int=True)
        x = self.static_mul_div(x, mul=level_downsamples[level])
        return x

    def static_double_div_int_mul_patch(self, x, patch_size):
        x = self.static_mul_div(self.static_mul_div(x, div=patch_size, as_int=True), mul=patch_size)
        return x

    def static_calculate_point_patch_level(self, x, level_downsamples, level, patch_size, point_pixel=0):
        x = self.static_double_mul_div_int_mul_level(x, level_downsamples, level, point_pixel)
        x = self.static_double_div_int_mul_patch(x, patch_size)
        return x

    def static_calculate_point_patch_level_array(self, x, level_downsamples, level_array, patch_size, point_pixel=0):
        for i in range(len(level_array)):
            x[i] = self.static_double_mul_div_int_mul_level(x[i], level_downsamples, level_array[i], point_pixel)
        x = self.static_double_div_int_mul_patch(x, patch_size)
        return x

    def static_detect_point_exist(self, point_array: list, x, y):
        number_point = point_array.count([x, y])
        self.static.record("static_detect_point_exist", self.record_status)
        return number_point

    def static_calculate_area_point_single(self, point, level, level_point_array, area_location, detect_location,
                                           patch_size):
        [id_x, id_y] = point
        if [id_x, id_y, level] not in detect_location:
            number_point = self.static_detect_point_exist(level_point_array, id_x, id_y)
            detect_location.append([id_x, id_y, level])
            area_location.append([number_point, id_x, id_y, level, patch_size])
        self.static.record("static_detect_point_exist", self.record_status)
        return detect_location, area_location

    def static_calculate_area_point(self, point_array, patch_size, level_downsamples, level_array=None, level=None):
        if level_array is not None:
            use_array = True
            level_point_array = self.static_calculate_point_patch_level_array(point_array, level_downsamples,
                                                                              level_array, patch_size)
        elif level is not None:
            use_array = False
            level_point_array = self.static_calculate_point_patch_level(point_array, level_downsamples, level,
                                                                        patch_size)
        else:
            raise ExistError("level")
        area_location = []
        detect_location = []
        for i in range(len(level_point_array)):
            if use_array is True:
                level = level_array[i]
            detect_location, area_location = self.static_calculate_area_point_single(level_point_array[i], level,
                                                                                     level_point_array,
                                                                                     area_location, detect_location,
                                                                                     patch_size)
