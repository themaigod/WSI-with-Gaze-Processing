import openslide
import cv2
import numpy as np
import os
import random
import itertools
from scipy.spatial.distance import pdist
from scipy.stats import entropy
from tool.Error import (RegisterError, ModeError, ExistError, OutBoundError, NoWriteError)
from tool.manager import (Static, Manager, Inner, SingleManager)
from tool.dataProcesser import all_reader
from config.config import Config


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
        self.config = Config  # 导入config参数

    def inner_register(self):  # 类内函数注册
        self.inner.register(self.__init__)
        self.inner.register(self.inner_register)
        self.inner.register(self.record)
        self.inner.register(self.func_run)
        self.inner.register(self.inner_func_run)
        self.inner.register(self.func_show)
        self.inner.register(self.func_del)
        self.func_manager.register(self.detect_background)
        self.func_manager.register(self.detect_point_num)
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

    def func_run(self, func_name, inputs, func_type=0, input_mode=0):
        # 在类内运行指定函数，func_type=0，使用函数名，（即函数指针），如函数定义为def a()，则func_name=a
        # func_type=1，未达到预想功能不推荐使用，未来再考虑改良
        # input_mode=0，输入参数应包装在tuple里，如(parameter_1, parameter_2, ...)
        # input_mode=1，输入参数应包装在dict里，如{parameter_1_name: parameter_1, parameter_2_name: parameter_2, ...}
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

    def detect_background(self, img, point_array):  # 检测背景，使用请重写
        if img is None or point_array is None:
            raise ExistError("parameter in detect_background")
        self.record("detect_background", self.record_status)

    def detect_point_num(self):  # 检测区域内视点数量，判断是否需要被舍弃，使用请重写
        self.record("detect_point_num", self.record_status)

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

    def read_image(self):  # 读取图像，使用请重写
        self.record("read_image", self.record_status)

    def read_image_area(self):  # 读取图像区域，使用请重写
        self.record("read_image_area", self.record_status)

    def set_label(self):  # 获得标签，使用请重写
        self.record("set_label", self.record_status)

    def process(self):  # 获得标签，使用请重写
        self.record("process", self.record_status)

    def save(self):  # 保存数据，使用请重写
        self.record("save", self.record_status)

    def get_order(self, mode=0):  # 获得调用函数详情
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
            [name, point_array, level_array, path, max_num, image_label] = all_reader(erc_path)  # 获取数据
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
        # 将erc文件存的路径转化为给定文件夹的路径
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
        # 将erc文件存的路径转化为给定文件夹的路径
        # 批量转化，调用name2path
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
            name_list = name_list.copy()
            point_list = point_list.copy()
            level_list = level_list.copy()
            path_list = path_list.copy()
            image_label = image_label.copy()
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

    def re_level_single(self, level):  # 将储存的level转化为openslide的level
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
        # 将储存的level转化为openslide的level
        # 批量转化，调用re_level_single
        # static版本
        for v in range(len(level_array)):
            for w in range(len(level_array[v])):
                level_array[v][w] = self.re_level_single(level_array[v][w])
        self.level_manager.record("static_re_level", self.record_status)
        return level_array

    def inner_re_level(self):
        # 将储存的level转化为openslide的level
        # 批量转化，调用re_level_single
        # inner版本
        for v in range(len(self.level_array)):
            for w in range(len(self.level_array[v])):
                self.level_array[v][w] = self.re_level_single(self.level_array[v][w])
        self.level_manager.record("inner_re_level", self.record_status)


class DatasetRegularProcess(GetDataset):
    def inner_register(self):
        super().inner_register()
        self.static.register(self.static_transform_to_1)
        self.static.register(self.static_get_same_num)
        self.static.register(self.static_calculate_class_num)

    def static_transform_to_1(self, array):  # 将矩阵转化为总和为1的矩阵，该步骤预设为支持数据集划分比例总和不为1的情况，例如，(7, 2, 1)
        result = []
        for i in range(len(array)):
            result.append(array[i] / sum(array))
        self.static.record("transform_to_1", self.record_status)
        return result

    def static_get_same_num(self, array):  # 统计array中有重复的元素和位置
        same_location = []
        same_value = []
        array = np.array(array)
        array_no_repeat = list(set(array))
        for i in range(len(array_no_repeat)):
            location = np.argwhere(array == array_no_repeat[i])
            if len(location) != 1:
                same_value.append(array_no_repeat[i])
                same_location.append(location)
        self.static.record("get_same_num", self.record_status)
        return same_value, same_location

    def static_calculate_class_num(self, array: list, total_num: int):
        # 依照总数和比例矩阵划分得到每一类的数量，例如输入数据为（7，1.5，1.5）、100，得到的应该是[70, 15, 15]
        # 该函数生成会符合以下规则：
        # 最重要：比例相同，数量相同
        # 其次：可能出现剩余, 剩余需尽可能少
        # 调用了get_same_num和transform_to_1
        array_num = [-1 for _ in range(len(array))]
        same_value, same_location = self.static_get_same_num(array)
        ratio = self.static_transform_to_1(array)
        count_class = 0
        count_num = 0
        class_num = len(array)
        for i in range(len(same_location)):  # 先划分有比例相同的
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
        if count_num != total_num:  # 如果有剩余单个的类
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

    def static_random_index(self, total_num: int):  # 生成固定范围（0，total_num)的不重复的随机索引值的list
        random_list = list(range(total_num))
        random.shuffle(random_list)
        self.static.record("static_random_index", self.record_status)
        return random_list

    def static_random_class_index(self, array: list, total_num: int):
        # 依照总数和比例矩阵划分得到属于每一类的索引值矩阵，例如输入数据为（1， 1）、6，得到的可能是[[6,1,3], [2,5,4]]
        # 调用了static_calculate_class_num和static_random_index
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

    def read_slide(self, path):  # 根据路径得到slide
        return openslide.OpenSlide(path)

    def read_image(self, slide=None, level=3):  # 读取特定level下的图像, 一般用于获得背景
        if slide is None:
            raise ExistError("slide")
        [length_of_img, height_of_img] = slide.level_dimensions[level]
        img = np.array(slide.read_region([0, 0], level, [length_of_img, height_of_img]))[:, :, :3]
        self.static.record("read_image", self.record_status)
        return img, level

    def static_resize_patch(self, img=None, patch_size=None):  # 根据patch_size改变img大小
        # 重新思考了这部分的作用，本来之前是用于在level_img=level的情况下使用，现在二者分开，好像意义不大，不如直接将level_img较高的值简便
        # 暂时舍去
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

    def static_bgr2gray(self, img):  # 转灰度图像
        # 注意opencv读入的是bgr图像
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.static.record("static_bgr2gray", self.record_status)
        return img

    def static_thresh(self, img=None):  # 二值化，方法otsu
        ret2, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.static.record("static_thresh", self.record_status)
        return img

    def static_thresh_patch(self, img):  # 转灰度然后二值化
        # img = self.static_resize_patch(img, patch_size) # 舍去理由见定义
        img = self.static_bgr2gray(img)
        img = self.static_thresh(img)
        self.static.record("static_thresh_patch", self.record_status)
        return img

    # def static_read_slide(self, path):   # 简化成之前的read_slide了
    #     slide = openslide.OpenSlide(path)
    #     level_downsamples = slide.level_downsamples
    #     self.static.record("static_thresh_patch", self.record_status)
    #     return slide, level_downsamples

    def static_mul_div(self, a_list, mul=None, div=None, as_int=False, to_np=False):
        # list与标量的乘法、除法、取整、转ndarray操作汇总
        # 考虑到速度和list的特殊性，转化成ndarray操作
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
        # 坐标x从point_pixel（level值）转到level（level值），例如从0转为2，x=（1000，1000），输出为（250， 250）
        x = self.static_mul_div(x, mul=level_downsamples[point_pixel], div=level_downsamples[level], as_int=True)
        return x

    def static_double_div_int_mul_patch(self, x, patch_size, level_downsamples, level):
        # 同一patch上的注视点坐标转为patch的左上角坐标（坐标的level为level），然后转回level=0的坐标
        x = self.static_mul_div(self.static_mul_div(x, div=patch_size, as_int=True), mul=patch_size)
        x = self.static_mul_div(x, mul=level_downsamples[level])
        return x

    def static_calculate_point_patch_level(self, x, level_downsamples, level, patch_size, point_pixel=0):
        # 同一patch上的注视点坐标转为patch的左上角坐标
        # 调用了static_double_mul_div_int_mul_level和static_double_div_int_mul_patch分步实现
        x = self.static_double_mul_div_int_mul_level(x, level_downsamples, level, point_pixel)
        x = self.static_double_div_int_mul_patch(x, patch_size, level_downsamples, level)
        return x

    def static_calculate_point_patch_level_start_point(self, x, level_downsamples, level, patch_size,
                                                       start_point=(0, 0), point_pixel=0):
        # 同一patch上的注视点坐标转为patch的左上角坐标，与static_calculate_point_patch_level区别在于
        # 支持指定取patch起始点
        # 如果对重叠取patch有需求，该函数能派上用场
        x = self.static_double_mul_div_int_mul_level(x, level_downsamples, level, point_pixel)
        start_point = self.static_double_mul_div_int_mul_level(start_point, level_downsamples, level, point_pixel)
        idx = [x[0] - start_point[0], x[1] - start_point[1]]
        idx = self.static_double_div_int_mul_patch(idx, patch_size, level_downsamples, level)
        x = [idx[0] + start_point[0], idx[1] + start_point[1]]
        return x

    def static_calculate_point_patch_level_array(self, x, level_downsamples, level_array, patch_size, point_pixel=0):
        # 对储存成list的所有注视点坐标进行：同一patch上的注视点坐标转为patch的左上角坐标
        x = x.copy()
        for i in range(len(level_array)):
            x[i] = self.static_double_mul_div_int_mul_level(x[i], level_downsamples, level_array[i], point_pixel)
            x[i] = self.static_double_div_int_mul_patch(x[i], patch_size, level_downsamples, level_array[i])
        return x

    def static_transform_pixel(self, point, level_downsamples, patch_size, level, point_pixel=0):
        # 将patch的索引转为level=point_pixel的坐标
        # 比方说，patch的索引可能为（0，0）....（m，n）
        # 那么例如，索引为（2，4），patch_size=224, level=1
        # 坐标为（2*224*2，4*224*2）
        point = self.static_mul_div(point, mul=patch_size)
        point = self.static_double_mul_div_int_mul_level(point, level_downsamples, point_pixel, level)
        return point

    def static_detect_point_exist(self, point_array: list, x, y):
        # 计算该坐标出现点坐标矩阵次数
        number_point = point_array.count([x, y])
        self.static.record("static_detect_point_exist", self.record_status)
        return number_point

    def static_calculate_area_point_single(self, point, level, level_point_array, area_location, detect_location,
                                           patch_size):
        # 从基于注视点的point_array生成基于patch的矩阵
        # 这是处理单个点，处理整个point_array的是static_calculate_area_point
        # single结尾的都是处理单个数据，往往还有一个能够遍历所有数据的函数，该函数与去掉single的函数名相似
        # 生成数据分别为area_location单个元素和detect_location单个元素
        # area_location单个元素结构为[注视点数量, x坐标, y坐标, level, patch_size]
        # detect_location单个元素结构为[x坐标, y坐标, level, patch_size]
        # area_location,detect_location是之后处理数据主要数据结构！！！
        area_location = area_location.copy()
        detect_location = detect_location.copy()
        [id_x, id_y] = point
        if [id_x, id_y, level, patch_size] not in detect_location:
            number_point = self.static_detect_point_exist(level_point_array, id_x, id_y)
            detect_location.append([id_x, id_y, level, patch_size])
            area_location.append([number_point, id_x, id_y, level, patch_size])
        self.static.record("static_detect_point_exist", self.record_status)
        return detect_location, area_location

    def static_calculate_area_point(self, point_array, patch_size, level_downsamples, level_array=None, level=None):
        # 从基于注视点的point_array生成基于patch的矩阵
        # 生成数据分别为area_location单个元素和detect_location单个元素
        # area_location单个元素结构为[注视点数量, x坐标, y坐标, level, patch_size]
        # detect_location单个元素结构为[x坐标, y坐标, level, patch_size]
        # area_location,detect_location是之后处理数据主要数据结构！！！
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
        return detect_location, area_location

    def static_point_limit(self, point, limit, index=0, equal=False):
        # 一个比较函数
        # 比较的是point的某一维度的坐标与limit（标量）的结果
        if equal is False:
            return point[index] > limit
        elif equal is True:
            return point[index] >= limit
        else:
            raise ModeError("equal")

    def static_gain_point_out_of_limits(self, point_array, point, mode=0):  # mode=0为start point，mode=1为end point
        # 对point array的每个point，检测是否在范围之外
        # start_point和end_point区别是
        # 如果某一维度小于start point即为False
        # 如果某一维度大于end point即为False
        result = []
        if mode == 0:
            for i in range(len(point_array)):
                result.append(
                    self.static_point_limit(point_array[i], point[0], index=0, equal=True) and self.static_point_limit(
                        point_array[i], point[1], index=1, equal=True))
        elif mode == 1:
            for i in range(len(point_array)):
                result.append(
                    not (self.static_point_limit(point_array[i], point[0], index=0) or self.static_point_limit(
                        point_array[i], point[1], index=1)))
        else:
            raise ModeError("in limit")
        return result

    def static_list_compare(self, list_a, list_b, mode=0):  # mode=0 list对位且运算， mode=1 list对位或运算
        list_result = []
        if mode == 0:
            for i in range(len(list_a)):
                list_result.append(list_a[i] and list_b[i])
        elif mode == 1:
            for i in range(len(list_a)):
                list_result.append(list_a[i] or list_b[i])
        else:
            raise ModeError("in list compare")
        return list_result

    def static_gain_point_out_of_area(self, point_array, start_point: list or tuple = (0, 0), area_size=None):
        # 检测是否超出由start point和area size指定的范围
        if area_size is None:
            result = self.static_gain_point_out_of_limits(point_array, start_point, 0)
        else:
            result_start = self.static_gain_point_out_of_limits(point_array, start_point, 0)
            end_point = [start_point[0] + area_size[0], start_point[1] + area_size[1]]
            result_end = self.static_gain_point_out_of_limits(point_array, end_point, 1)
            result = self.static_list_compare(result_start, result_end, 0)
        return result

    def static_del_list_position_index(self, list_a: list, index_list: list, reverse=False):
        # 根据index_list提供的索引删除list a里的元素
        # reverse代表是否去反，结果是一样的，但实现细节有些许不同，可能导致速度差异
        # 目前实现的检测类函数得到的结果都不调用该函数进行删除操作，如有兴趣，可以重写检测类函数使得符合该函数需要的结果，或许有助于提升速度
        list_a = list_a.copy()
        if reverse is False:
            index_list.sort()
            for i in range(len(index_list)):
                list_a.pop(index_list[i] - i)
        elif reverse is True:
            index_list.sort(reverse=reverse)
            for i in range(len(index_list)):
                list_a.pop(index_list[i])
        return list_a

    def static_del_list_position_bool(self, list_a: list, bool_list):
        # 根据bool_list删除list a里的元素
        # 如果某位置bool_list是False，删除该位置的list a元素
        # 目前实现的检测类函数得到的结果都需要调用该函数进行删除操作，如有兴趣，可以实现下基于index的
        # 该函数功能可以实现像index版本的函数内的reverse控制，可能有助于速度
        # 但因为实现细节稍微更复杂了点，暂未实现，如有兴趣，欢迎提供该版本
        list_b = list_a.copy()
        for i in reversed(range(len(list_b))):
            if bool_list[i] is False:
                list_b.pop(i)
        return list_b

    def static_separate_list_position_bool(self, list_a: list, bool_list):
        # 类似上面一个函数
        # 该函数功能是根据bool list拆分成两个函数
        list_b = []
        list_c = list_a.copy()
        for i in reversed(range(len(list_c))):
            if bool_list[i] is False:
                list_b.insert(0, list_c[i])
                list_c.pop(i)
        return list_c, list_b

    def static_calculate_particular_area_point(self, point_array, patch_size, level_downsamples, start_point=(0, 0),
                                               area_size=None, level_array=None, level=None):
        # 与static_calculate_area_point类似
        # 支持start_point, area_size
        result = None
        if start_point != (0, 0) or (start_point == (0, 0) and (area_size is not None)):
            result = self.static_gain_point_out_of_area(point_array, start_point, area_size)
            point_array = self.static_del_list_position_bool(point_array, result)
        if level_array is not None:
            use_array = True
            if result is not None:
                level_array = self.static_del_list_position_bool(level_array, result)
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
        return detect_location, area_location

    def detect_point_num(self, area_location=None, detect_location=None, threshold=0):
        # 根据注视点数量阈值筛选区域
        if area_location is None:
            raise ExistError("area_location")
        if detect_location is None:
            after_processing = self.static_filter_by_threshold_area(area_location, threshold)
            return after_processing
        else:
            after_processing_area, after_processing_detect = self.static_filiter_by_threshold_area_detect(area_location,
                                                                                                          detect_location,
                                                                                                          threshold)
            return after_processing_area, after_processing_detect

    def static_filiter_by_threshold_area_detect(self, area_location, detect_location, threshold):
        after_processing_area = area_location.copy()
        after_processing_detect = detect_location.copy()
        if threshold != 0:
            for i in range(len(area_location)):
                if area_location[i][0] < threshold:
                    after_processing_area.pop(i)
                    after_processing_detect.pop(i)
        return after_processing_area, after_processing_detect

    def static_filter_by_threshold_area(self, area_location, threshold):
        after_processing = area_location.copy()
        if threshold != 0:
            for i in reversed(range(len(area_location))):
                if area_location[i][0] < threshold:
                    after_processing.pop(i)
        return after_processing

    def detect_background_single(self, img, x, y, level_img, level_downsamples, reverse=True):
        # reverse=True时, 说明img长宽是颠倒, 往往因为PIL图像与OPENCV图像（numpy array）转换的关系
        # 对单点检测是否不是背景
        self.static_double_mul_div_int_mul_level([x, y], level_downsamples, level_img)
        if reverse is True:
            idx = y
            idy = x
        elif reverse is False:
            idx = x
            idy = y
        else:
            raise ExistError("reverse")
        if idx >= img.shape[0] or idy >= img.shape[1]:
            foreground = None
        else:
            if img[idx, idy] == 255:
                foreground = False
            else:
                foreground = True
        return foreground

    def detect_background(self, img, point_array, level_img=None, level_downsamples=None, position=(1, 2),
                          reverse=True):
        # 对point array检测背景
        result = []
        for i in range(len(point_array)):
            foreground = self.detect_background_single(img, point_array[i][position[0]], point_array[i][position[1]],
                                                       level_img, level_downsamples, reverse)
            result.append(foreground)
        return result

    def detect_edge_single(self, point=None, level=3, patch_size=None, level_downsamples=None, level_dimensions=None,
                           start_point=(0, 0), area_size=None, reformat=True):
        # 对单点检测是否超出边界
        # 该函数可能因耗时久以及之前操作隐含边界检测而不实用
        # 但该函数优势是检测是准确严格的，之前操作隐含边界检测可能存在缺陷
        result = True
        if reformat is True:
            point = self.static_calculate_point_patch_level_start_point(point, level_downsamples, level, patch_size,
                                                                        start_point, 0)
        if start_point != (0, 0):
            if area_size is None:
                end_point = [level_dimensions[level][0] - 1, level_dimensions[level][1] - 1]
            else:
                end_point = [start_point[0] + area_size[0], start_point[1] + area_size[1]]
                if end_point[0] >= level_downsamples[level][0] or end_point[1] >= level_downsamples[level][1]:
                    raise OutBoundError("area")
            end_point_reformat = self.static_calculate_point_patch_level_start_point(end_point, level_downsamples,
                                                                                     level, patch_size,
                                                                                     start_point, 0)
            patch_size_pixel_0 = self.static_mul_div(patch_size, mul=level_downsamples[level], div=level_downsamples[0],
                                                     as_int=True)
            if (end_point_reformat[0] + patch_size_pixel_0 - 1) != end_point[0]:
                end_point[0] = end_point_reformat[0] - patch_size_pixel_0
            else:
                end_point[0] = end_point_reformat[0]
            if (end_point_reformat[1] + patch_size_pixel_0 - 1) != end_point[1]:
                end_point[1] = end_point_reformat[1] - patch_size_pixel_0
            else:
                end_point[1] = end_point_reformat[1]
            result_start_x = self.static_point_limit(point, start_point[0], index=0, equal=True)
            result_start_y = self.static_point_limit(point, start_point[1], index=1, equal=True)
            result_start = result_start_x and result_start_y
            result_end = not (self.static_point_limit(point, end_point[0], index=0) or self.static_point_limit(
                point, end_point[1], index=1))
            result = result_start and result_end
        elif start_point == (0, 0):
            if area_size is not None:
                end_point = [start_point[0] + area_size[0], start_point[1] + area_size[1]]
                if end_point[0] >= level_downsamples[level][0] or end_point[1] >= level_downsamples[level][1]:
                    raise OutBoundError("area")
            else:
                end_point = [level_dimensions[level][0] - 1, level_dimensions[level][1] - 1]
            end_point_reformat = self.static_calculate_point_patch_level_start_point(end_point, level_downsamples,
                                                                                     level, patch_size,
                                                                                     start_point, 0)
            patch_size_pixel_0 = self.static_mul_div(patch_size, mul=level_downsamples[level], div=level_downsamples[0],
                                                     as_int=True)
            if (end_point_reformat[0] + patch_size_pixel_0 - 1) != end_point[0]:
                end_point[0] = end_point_reformat[0] - patch_size_pixel_0
            else:
                end_point[0] = end_point_reformat[0]
            if (end_point_reformat[1] + patch_size_pixel_0 - 1) != end_point[1]:
                end_point[1] = end_point_reformat[1] - patch_size_pixel_0
            else:
                end_point[1] = end_point_reformat[1]
            result = not (self.static_point_limit(point, end_point[0], index=0) or self.static_point_limit(
                point, end_point[1], index=1))
        return result

    def detect_edge(self, point_array: list = None, level_array=None, level: type(None) or int = None, patch_size=None,
                    level_downsamples=None,
                    level_dimensions=None, start_point=(0, 0), area_size=None, reformat=True):
        # point array输入area_location时，reformat输入True，如果输入的是detect_location或者point array， reformat应设为False
        # 存在level_array时, 使用level_array, 否则考虑使用level. level如为None, point_array类型为area_location或detect_location, 使用point_array自带的level
        # 所以要使用area_location或detect_location带的level, 一定要将level设为None！！！
        # 对point_array检测是否超出边界
        result = []
        if level_array is None:
            if level is None and len(point_array[0]) > 2:
                level_array = [j[-2] for j in point_array]
            elif level is not None:
                level_array = [level for _ in range(len(point_array))]
            else:
                raise ExistError("level or level_array")
        for i in range(len(point_array)):
            if patch_size is None:
                patch_size = point_array[i][-1]
            if reformat is True:
                result_single = self.detect_edge_single(point_array[i][1:3], level_array[i], patch_size,
                                                        level_downsamples,
                                                        level_dimensions, start_point, area_size, reformat)
            elif reformat is False:
                result_single = self.detect_edge_single(point_array[i][0:2], level_array[i], patch_size,
                                                        level_downsamples,
                                                        level_dimensions, start_point, area_size, reformat)
            else:
                raise ModeError("reformat")
            result.append(result_single)
        return result

    def static_gain_level_positive(self, area_location, mode=0):
        # 获取area_location使用的level：detect_level
        # 获取area_location使用的level及数量：result，例如：result：[[0, 500], [1,30]]
        result = []
        detect_level = []
        for i in range(len(area_location)):
            if area_location[i][3] not in detect_level:
                detect_level.append(area_location[i][3])
                result.append([area_location[i][3], 1])
            else:
                index = detect_level.index(area_location[i][3])
                result[index][1] += 1
        if mode == 0:
            return result
        elif mode == 1:
            return detect_level
        elif mode == 2:
            return detect_level, result
        else:
            raise ModeError("in static_gain_level_positive")

    def static_all_list(self, level_dimensions, detect_level, level_downsamples, patch_size, mode=0, location_type=0):
        # 根据detect_level提供的level获得在这些level上的patch，包括detect location[location_type = 0]、area location[location_type = 1]两种形式，注意area location
        # 形式的时候，注视点数量设为0。location_type = 2时提供一种不完备的形式, 即在代表每个patch元素中，同时储存detect location和area location
        # 该location_type输出可以由static_separate_all_list_location_type_2转成正常的detect location和area location\\该方案已废弃
        # 更好的解决方案可能是生成area location，再经由static_area_location2detect_location生成detect location
        # mode指导返回数据
        # mode=0，返回的形式是[list1, list2, list3...], list1是level=detect_level[0]时所有的patch，以此类推
        # mode=1，返回标准的location，如，[location1, location2, ...], 包含所有的patch
        # mode=2, 同时返回mode=0和mode=1的结果
        result = []
        result_optional = []
        for i in range(len(detect_level)):
            point = [level_dimensions[detect_level[i]][0] - 1, level_dimensions[detect_level[i]][0] - 1]
            point_now_level = self.static_double_mul_div_int_mul_level(point, level_downsamples, detect_level[i], 0)
            point_level_patch = self.static_mul_div(point_now_level, div=patch_size, as_int=True)
            point_now_level_patch = self.static_mul_div(point_level_patch, mul=patch_size)
            point_now_level_end = [0, 0]
            if (point_now_level_patch[0] + patch_size - 1) != point_now_level:
                point_now_level_end[0] = point_now_level_patch[0] - patch_size
            else:
                point_now_level_end[0] = point_now_level_patch[0]
            if (point_now_level_patch[1] + patch_size - 1) != point_now_level:
                point_now_level_end[1] = point_now_level_patch[1] - patch_size
            else:
                point_now_level_end[1] = point_now_level_patch[1]
            loop = self.static_mul_div(point_now_level_end, div=patch_size, as_int=True)
            result_level = []
            for j in range(loop[0]):
                for k in range(loop[1]):
                    point_dynamic = self.static_transform_pixel([j, k], level_downsamples, patch_size, detect_level[i],
                                                                0)
                    if location_type == 0:
                        location = [0, point_dynamic[0], point_dynamic[1], detect_level[i], patch_size]
                    elif location_type == 1:
                        location = [point_dynamic[0], point_dynamic[1], detect_level[i], patch_size]
                    elif location_type == 2:
                        location = [[0, point_dynamic[0], point_dynamic[1], detect_level[i], patch_size],
                                    [point_dynamic[0], point_dynamic[1], detect_level[i], patch_size]]
                    else:
                        raise ModeError("location_type")
                    if mode == 0:
                        result_level.append(location)
                    elif mode == 1:
                        result.append(location)
                    elif mode == 2:
                        result_level.append(location)
                        result.append(location)
                    else:
                        raise ModeError("in static_all_list")
            if mode == 0:
                result.append(result_level)
            elif mode == 2:
                result_optional.append(result_level)
        if mode == 0 or mode == 1:
            return result
        elif mode == 2:
            return result, result_optional
        else:
            raise ModeError("in static_all_list")

    def static_separate_all_list_location_type_2(self, all_list):
        # 将static_all_list的location type=2的得到的类型转成常见的area_location, detect_location
        area_location = []
        detect_location = []
        for i in range(len(all_list)):
            area_location.append(all_list[i][0])
            detect_location.append(all_list[i][1])
        return area_location, detect_location

    def static_area_location2detect_location(self, area_location):
        # area_location转detect_location
        detect_location = [area_location[i][1:] for i in range(len(area_location))]
        return detect_location

    def static_detect_exist_point_array(self, all_list, detect_location, all_type=0, location_type=0):
        # 检测all_list中的元素是否存在于detect_location
        # 返回值是长度与all_list相同的list，元素为bool值，False表示该位置的元素存在于detect_location
        if all_type == 1:
            all_list = [i[1:] for i in all_list]
        if location_type == 1:
            detect_location = [j[1:] for j in detect_location]
        result = [True for _ in range(len(all_list))]
        for k in range(len(all_list)):
            if all_list[k] in detect_location:
                result[k] = False
        return result

    def static_all_list_point_num_repair(self, all_list, area_location):
        # 将area location格式的all list的注视点数量根据area_location进行修正
        all_list = all_list.copy()
        detect_area_location = [area_location[i][1:] for i in range(len(area_location))]
        for j in range(len(all_list)):
            if all_list[j][1:] in detect_area_location:
                index = detect_area_location.index(all_list[j][1:])
                all_list[j][0] = area_location[index][0]
        return all_list

    def set_label(self, x=None, y=None, level=None, patch_size=None, detect_location=None, location_type=0, mode=0):
        # 依据是否包含在detect_location获得标签
        # detect_location一般是确认包含注视点的patch集合
        # 这样，1代表被注视过（或者说包含注视点），0则没被注视过
        if mode == 0:
            point = x
            if location_type == 1:
                detect_location = [j[1:] for j in detect_location]
        elif mode == 1:
            point = x
            if location_type == 0:
                detect_location = [j[1:] for j in detect_location]
            elif location_type == 1:
                detect_location = [j[1:-1] for j in detect_location]
        elif mode == 2:
            if location_type == 1:
                if patch_size is None:
                    point = [x, y, level]
                    detect_location = [j[1:-1] for j in detect_location]
                else:
                    point = [x, y, level, patch_size]
                    detect_location = [j[1:] for j in detect_location]
            elif location_type == 0:
                if patch_size is None:
                    point = [x, y, level]
                    detect_location = [j[:-1] for j in detect_location]
                else:
                    point = [x, y, level, patch_size]
            else:
                raise ModeError("in location type")
        else:
            raise ModeError("in set label")
        if point in detect_location:
            label = 1
        else:
            label = 0
        return label

    def static_distance_euclidean(self, point1, point2):  # 欧氏距离 两点的计算
        s = 0
        for i in range(len(point1)):
            s += pow(point1[i] - point2[i], 2)
        s = pow(s, 0.5)
        return s

    def static_distance_minkowsk(self, point1, point2, p):  # 闵式距离 两点的计算
        s = 0
        for i in range(len(point1)):
            s += pow(point1[i] - point2[i], p)
        s = pow(s, 1 / p)
        return s

    def static_distance_standardized_euclidean(self, point1, point2):  # 标准化的欧氏距离 两点的计算
        s = 0
        sk = [0 for _ in range(len(point1))]
        for i in range(len(point1)):
            sk[i] = pow(point1[i] / 2 - point2[i] / 2, 2) / (2 - 1)
        for i in range(len(point1)):
            if sk[i] != 0:
                s += pow(point1[i] - point2[i], 2) / sk[i]
        s = pow(s, 0.5)
        return s

    def static_distance_KL(self, point1, point2):  # KL散度 两点的计算
        return entropy(point1, point2)

    def static_distance_euclidean_group(self, point_group1, to_list=True):
        # 欧几里得距离 对一群点互相之间计算
        # 输出shape = (len(point_group1), len(point_group1))
        point_group1 = np.array(point_group1)
        point_group2 = point_group1.copy()
        distance = np.reshape(np.sum(point_group1 ** 2, axis=1), (point_group1.shape[0], 1)) + np.sum(point_group2 ** 2,
                                                                                                      axis=1) - 2 * point_group1.dot(
            point_group2.T)
        if to_list is True:
            distance = distance.tolist()
        return distance

    def static_distance_minkowsk_group(self, point_group1, p, to_list=True, transform=True):
        # 闵氏距离 对一群点互相之间计算
        # transform之前，shape = len(point_group1) * len(point_group1)
        point_group1 = np.array(point_group1)
        point_group2 = point_group1.copy()
        vector1, vector2 = self.static_create_group2vector(point_group1, point_group2)
        distance = np.linalg.norm(vector1 - vector2, ord=p, axis=1)
        if transform is True:
            distance = distance.reshape((len(point_group1), len(point_group1)))
        if to_list is True:
            distance = distance.tolist()
        return distance

    def static_create_group2vector(self, point_group1, point_group2):
        vector1 = np.zeros((len(point_group1) * len(point_group1), point_group1.shape[1]))
        vector2 = np.zeros((len(point_group1) * len(point_group1), point_group1.shape[1]))
        for i in range(len(point_group1)):
            vector1[i * len(point_group2): (i + 1) * len(point_group2), :] = point_group1[i]
        for j in range(len(point_group1)):
            vector2[j * len(point_group2): (j + 1) * len(point_group2), :] = point_group2[:]
        return vector1, vector2

    def static_distance_standardized_euclidean_group(self, point_group1, to_list=True, transform=True):
        # 标准化欧几里得距离 对一群点互相之间计算
        # transform之前，shape = len(point_group1) * len(point_group1)
        point_group1 = np.array(point_group1)
        distance_group = pdist(point_group1, "seuclidean", V=None)
        distance = np.zeros((len(point_group1), len(point_group1)))
        index = 0
        for i in range(len(point_group1)):
            for j in range(len(point_group1)):
                if i < j:
                    distance[i, j] = distance_group[index]
                    distance[j, i] = distance_group[index]
                    index += 1
        if transform is True:
            distance = distance.reshape((len(point_group1), len(point_group1)))
        if to_list is True:
            distance = distance.tolist()
        return distance

    def static_distance_KL_group(self, point_group1, to_list=True, transform=True):
        # kl散度 对一群点互相之间计算
        # transform之前，shape = len(point_group1) * len(point_group1)
        point_group1 = np.array(point_group1)
        point_group2 = point_group1.copy()
        vector1, vector2 = self.static_create_group2vector(point_group1, point_group2)
        distance = np.array(entropy(vector1, vector2, axis=1))
        if transform is True:
            distance = distance.reshape((len(point_group1), len(point_group1)))
        if to_list is True:
            distance = distance.tolist()
        return distance

    def static_distance_euclidean_groups(self, point_group1, point_group2, to_list=True):
        # 欧几里得距离 对两群点互相之间计算
        point_group1 = np.array(point_group1)
        point_group2 = np.array(point_group2)
        distance = np.reshape(np.sum(point_group1 ** 2, axis=1), (point_group1.shape[0], 1)) + np.sum(point_group2 ** 2,
                                                                                                      axis=1) - 2 * point_group1.dot(
            point_group2.T)
        if to_list is True:
            distance = distance.tolist()
        return distance

    def static_distance_minkowsk_groups(self, point_group1, point_group2, p, to_list=True, transform=True):
        # 闵氏距离 对两群点互相之间计算
        point_group1 = np.array(point_group1)
        point_group2 = np.array(point_group2)
        vector1, vector2 = self.static_create_group2vector(point_group1, point_group2)
        distance = np.linalg.norm(vector1 - vector2, ord=p, axis=1)
        if transform is True:
            distance = distance.reshape((len(point_group1), len(point_group2)))
        if to_list is True:
            distance = distance.tolist()
        return distance

    def static_distance_standardized_euclidean_groups(self, point_group1, point_group2, to_list=True):
        # 标准化欧几里得距离 对两群点互相之间计算
        point_group1 = np.array(point_group1)
        point_group2 = np.array(point_group2)
        distance_group = pdist(np.vstack([point_group1, point_group2]), "seuclidean", V=None)
        distance = np.zeros((len(point_group1), len(point_group2)))
        for i in range(len(point_group1)):
            for j in range(len(point_group2)):
                distance[i, j] = distance_group[((len(point_group1) + len(point_group2)) * i) // 2 + j - i - 1]
        if to_list is True:
            distance = distance.tolist()
        return distance

    def static_distance_KL_groups(self, point_group1, point_group2, to_list=True, transform=True):
        # kl散度 对两群点互相之间计算
        point_group1 = np.array(point_group1)
        point_group2 = np.array(point_group2)
        vector1, vector2 = self.static_create_group2vector(point_group1, point_group2)
        distance = np.array(entropy(vector1, vector2))
        if transform is True:
            distance = distance.reshape((len(point_group1), len(point_group2)))
        if to_list is True:
            distance = distance.tolist()
        return distance

    def static_calculate_distance(self, point1, point2, mode=0, point_type=0, p=2):
        # 计算点对点的距离
        # 根据mode选计算方式
        if point_type == 1:
            point1 = point1[1:]
            point2 = point2[1:]
        if mode == 0:
            distance = self.static_distance_euclidean(point1, point2)
        elif mode == 2:
            distance = self.static_distance_standardized_euclidean(point1, point2)
        elif mode == 3:
            distance = self.static_distance_minkowsk(point1, point2, p)
        else:
            raise ModeError("in static_calculate_distance")
        return distance

    def static_start2center_list(self, detect_location: list, level_downsamples):
        # detect_location位置同时支持area_location和detect_location
        # 转换detect_location中的坐标
        # 原先代表patch的坐标是左上角的位置，转换后是中心点
        detect_location_result = detect_location.copy()
        for i in range(len(detect_location)):
            patch_size = detect_location[i][-1]
            level = detect_location[i][-2]
            x = detect_location[i][-4]
            y = detect_location[i][-3]
            x = x - self.static_mul_div(patch_size, mul=level_downsamples[level], div=level_downsamples[0], as_int=True)
            y = y - self.static_mul_div(patch_size, mul=level_downsamples[level], div=level_downsamples[0], as_int=True)
            detect_location_result[i][-4] = x
            detect_location_result[i][-3] = y
        return detect_location_result

    def static_all_distance(self, all_center_list, location_type=(0, 0), mode=0, p=2):
        # 计算all_center_list所有patch互相之间距离
        # location_type[0]负责尾部处理，[1]区分area_location, detect_location, mode区分使用的算法
        # mode指定使用哪种距离
        if location_type[0] == 0:
            if location_type[1] == 0:
                location_array = [i[:-2] for i in all_center_list]
            elif location_type[1] == 1:
                location_array = [i[1:-2] for i in all_center_list]
            else:
                raise ExistError("location_type")
        elif location_type[0] == 1:
            if location_type[1] == 0:
                location_array = [i[:-1] for i in all_center_list]
            elif location_type[1] == 1:
                location_array = [i[1:-1] for i in all_center_list]
            else:
                raise ExistError("location_type")
        else:
            raise ExistError("location_type")
        if mode == 0:
            distance = self.static_distance_euclidean_group(location_array, to_list=False)
            # 输出是numpy ndarray， 如需要list, to list设为True或者使用默认
        elif mode == 1:
            distance = self.static_distance_minkowsk_group(location_array, p, to_list=False)
        elif mode == 2:
            distance = self.static_distance_standardized_euclidean_group(location_array, to_list=False)
        elif mode == 3:
            distance = self.static_distance_KL_group(location_array, to_list=False)
        else:
            raise ModeError(str(mode) + "in static_all_distance")
        return distance

    def static_transform_num2mark_single(self, x, mode=0):
        # 一般用于根据注视点数量转换成分值
        # 目前是等值转换，未来可以重写或添加mode的方式引入新的计算方法
        if mode == 0:
            x = x
        else:
            raise ModeError(str(mode) + "in static_transform_num2mark_single")
        return x

    def static_transform_num2mark(self, area_location):
        # 对整个area_location都进行根据注视点数量转换成分值
        area_location = area_location.copy()
        for i in range(len(area_location)):
            area_location[i][0] = self.static_transform_num2mark_single(area_location[i][0])
        return area_location

    def static_calculate_point_mark(self, mark1, mark2, mode=0):
        # 基于两点的分值计算两点间的权重值
        # 目前采用相加除以2的方式计算，未来可以重写或添加mode的方式引入新的计算方法
        if mode == 0:
            mark = (mark1 + mark2) / 2
        else:
            raise ModeError(str(mode) + "static_calculate_point_mark")
        return mark

    def static_calculate_point_group_mark(self, all_list, mode=0, to_list=True):
        # 不支持detect_location
        # 计算所有patch相互之间的权重值
        # mode为static_calculate_point_mark所需要的mode，用于改变计算方法
        result_list = np.zeros((len(all_list), len(all_list)))
        for i in range(len(all_list)):
            for j in range(len(all_list)):
                if i < j:
                    result_list[i, j] = self.static_calculate_point_mark(all_list[i][0], all_list[j][0], mode)
                    result_list[j, i] = result_list[i, j]
        if to_list is True:
            result_list.tolist()
        return result_list

    def static_calculate_point_group_distance_mark(self, result_list, distance, to_list=True):
        # 基于距离和权重值计算所有patch之间的连接对patch重要性的贡献
        result_list = np.array(result_list)
        distance = np.array(distance)
        result = result_list / distance
        if to_list is True:
            result.tolist()
        return result

    def static_calculate_all_point_get_mark(self, final_list, mode=0):
        # 对于所有patch，计算基于该patch对于其他patch的连接的贡献的总和（mode=0）或者平均值（mode=1）
        # 以此来衡量patch重要性
        if mode == 0:
            result = [0 for _ in range(len(final_list))]
        elif mode == 1:
            result = [float(0) for _ in range(len(final_list))]
        else:
            result = [0 for _ in range(len(final_list))]
        final_list = np.array(final_list)
        for i in range(len(final_list)):
            dynamic = 0
            for j in range(len(final_list)):
                if i < j:
                    dynamic += final_list[i, j]
                elif i > j:
                    dynamic += final_list[j, i]
            if mode == 0:
                result[i] = dynamic
            elif mode == 1:
                result[i] = dynamic / (len(final_list) - 1)
        return result

    def static_zero_follow_mark(self, mark_result, num, reverse=False):
        # 根据mark选择patch
        # 取前num个索引
        # 一般用于取negative标签（标签0）的样本的相关操作
        mark_index = [[i, mark_result[i]] for i in range(len(mark_result))]
        mark_index = sorted(mark_index, key=lambda s: s[1], reverse=reverse)
        if num > len(mark_index):
            num = len(mark_index)
        index = [mark_index[j][0] for j in range(num)]
        return index, num

    def static_zero_random_list(self, zero_list, num):
        # 随机取索引
        # 取前num个索引
        # 一般用于取negative标签（标签0）的样本的相关操作
        random_list = list(range(zero_list))
        random.shuffle(random_list)
        if num > 1:
            if num > len(random_list):
                num = len(random_list)
            index = random_list[:num]
        elif num == 1:
            index = [random_list[0]]
        else:
            raise OutBoundError("num " + str(num))
        return index, num

    def static_create_positive_marked_area_location(self, mark_result, all_area_location: list, area_location,
                                                    location_type=(0, 0)):
        marked_area_location = area_location.copy()
        if location_type[1] == 0:
            detect_location = [area_location[j][1:] for j in range(len(area_location))]
        elif location_type[1] == 1:
            detect_location = area_location.copy()
        else:
            raise ModeError(str(location_type[1]) + " in location_type[1]")
        if location_type[0] == 0:
            index = 1
        elif location_type[1] == 1:
            index = 0
        else:
            raise ModeError(str(location_type[0]) + " in location_type[0]")
        for i in range(len(all_area_location)):
            if all_area_location[i][index:] in detect_location:
                get_index = detect_location.index(all_area_location[i][index:])
                marked_area_location[get_index].insert(0, mark_result[i])
        return marked_area_location

    def static_create_negative_marked_area_location(self, mark_result, zero_area_location: list):
        zero_area_location = zero_area_location.copy()
        for i in range(len(zero_area_location)):
            zero_area_location[i].insert(0, mark_result[i])
        return zero_area_location

    def static_zero_list2zero_list_level(self, zero_list):
        result = []
        detect_level = []
        num_level = []
        for i in range(len(zero_list)):
            level = zero_list[i][-2]
            if level not in detect_level:
                detect_level.append(level)
                result.append([i])
                num_level.append([level, 1])
            else:
                index = detect_level.index(level)
                result[index].append(i)
                num_level[index][1] += 1
        return result, num_level, detect_level

    def static_get_num_index(self, detect_num, level):
        if level in detect_num:
            index = detect_num.index(level)
        else:
            index = None
        return index

    def static_get_num(self, num_level, index):
        if index is not None:
            num = num_level[index][1]
        else:
            num = 0
        return num

    def static_zero_list_num(self, zero_list, result_level, num, ratio, mode):
        zero_level_result, zero_num_level, zero_level = self.static_zero_list2zero_list_level(zero_list)
        if num is not None:
            detect_num = [num[i][0] for i in range(len(num))]
        else:
            detect_num = [None]
        if result_level is not None:
            detect_result_level = [result_level[i][0] for i in range(len(result_level))]
        else:
            detect_result_level = [None]
        result = []
        result_reduce = []
        for i in range(len(zero_level)):
            level = zero_level[i]
            index_result_level = self.static_get_num_index(detect_result_level, level)
            index_num = self.static_get_num_index(detect_num, level)
            now_num = self.static_get_num(num, index_num)
            result_level_num = self.static_get_num(result_level, index_result_level)
            if mode == 0:
                zero_true_num, reduce_zero = self.static_get_zero_num_mode0(now_num, result_level_num,
                                                                            zero_num_level[i][1])
                result.append([level, zero_true_num])
                result_reduce.append([level, reduce_zero])
            elif mode == 1:
                reduce_zero, zero_true_num = self.static_get_zero_num_mode1(now_num, ratio, result_level_num,
                                                                            zero_num_level[i][1])
                result.append([level, zero_true_num])
                result_reduce.append([level, reduce_zero])
            elif mode == 2:
                reduce_zero, zero_true_num = self.static_get_zero_num_mode2(now_num, ratio, zero_num_level[i][1])
                result.append([level, zero_true_num])
                result_reduce.append([level, reduce_zero])
            elif mode == 3:
                reduce_zero, zero_true_num = self.static_get_zero_num_mode3(now_num, ratio, zero_num_level[i][1])
                result.append([level, zero_true_num])
                result_reduce.append([level, reduce_zero])
            elif mode == 4:
                reduce_zero, zero_true_num = self.static_get_zero_num_mode4(now_num, ratio, result_level_num,
                                                                            zero_num_level[i][1])
                result.append([level, zero_true_num])
                result_reduce.append([level, reduce_zero])
            else:
                raise ModeError("in static_zero_list")
        return result, result_reduce, zero_level_result, zero_num_level, zero_level

    def static_get_zero_num_mode4(self, now_num, ratio, result_level_num, zero_num):
        if zero_num < result_level_num * ratio + now_num:
            zero_true_num = zero_num
            reduce_zero = result_level_num * ratio + now_num - zero_num
        else:
            zero_true_num = result_level_num * ratio + now_num
            reduce_zero = 0
        return reduce_zero, zero_true_num

    def static_get_zero_num_mode3(self, now_num, ratio, zero_num):
        if ratio is not None:
            if zero_num > (now_num * ratio):
                zero_true_num = now_num * ratio
                reduce_zero = 0
            else:
                zero_true_num = zero_num
                reduce_zero = (now_num * ratio) - zero_num
        else:
            if zero_num > now_num:
                zero_true_num = now_num
                reduce_zero = 0
            else:
                zero_true_num = zero_num
                reduce_zero = now_num - zero_num
        return reduce_zero, zero_true_num

    def static_get_zero_num_mode2(self, now_num, ratio, zero_num):
        if zero_num > (zero_num * ratio) + now_num:
            zero_true_num = zero_num * ratio + now_num
            reduce_zero = 0
        else:
            zero_true_num = zero_num
            reduce_zero = zero_num * ratio + now_num - zero_num
        return reduce_zero, zero_true_num

    def static_get_zero_num_mode1(self, now_num, ratio, result_level_num, zero_num):
        if zero_num < (result_level_num + now_num) * ratio:
            zero_true_num = zero_num
            reduce_zero = (result_level_num + now_num) * ratio - zero_num
        else:
            zero_true_num = (result_level_num + now_num) * ratio
            reduce_zero = 0
        return reduce_zero, zero_true_num

    def static_get_zero_num_mode0(self, now_num, result_level_num, zero_num):
        if zero_num < result_level_num + now_num:
            zero_true_num = zero_num
            reduce_zero = result_level_num + now_num - zero_num
        else:
            zero_true_num = result_level_num + now_num
            reduce_zero = 0
        return zero_true_num, reduce_zero

    def static_one_list_num(self, one_list, num, ratio, mode):
        level_result, num_level, one_level = self.static_zero_list2zero_list_level(one_list)
        if num is not None:
            detect_num = [num[i][0] for i in range(len(num))]
        else:
            detect_num = [None]
        result = []
        result_reduce = []
        for i in range(len(one_level)):
            level = one_level[i]
            index_num = self.static_get_num_index(detect_num, level)
            now_num = self.static_get_num(num, index_num)
            result_level_num = num_level[i][1]
            if mode == 0:
                true_num, reduce_one = self.static_get_one_num_mode0(now_num, result_level_num)
                result.append([level, true_num])
                result_reduce.append([level, reduce_one])
            elif mode == 1:
                true_num, reduce_one = self.static_get_one_num_mode1(now_num, ratio, result_level_num)
                result.append([level, true_num])
                result_reduce.append([level, reduce_one])
            elif mode == 2:
                true_num, reduce_one = self.static_get_one_num_mode2(now_num, ratio, result_level_num)
                result.append([level, true_num])
                result_reduce.append([level, reduce_one])
            elif mode == 3:
                true_num, reduce_one = self.static_get_one_num_mode3(now_num, ratio, result_level_num)
                result.append([level, true_num])
                result_reduce.append([level, reduce_one])
            else:
                raise ModeError("in static_zero_list")
        return result, result_reduce, level_result, num_level, one_level

    def static_get_one_num_mode3(self, now_num, ratio, result_level_num):
        if ratio is not None:
            if result_level_num > (now_num * ratio):
                true_num = now_num * ratio
                reduce_one = 0
            else:
                true_num = result_level_num
                reduce_one = (now_num * ratio) - result_level_num
        else:
            if result_level_num > now_num:
                true_num = now_num
                reduce_one = 0
            else:
                true_num = result_level_num
                reduce_one = now_num - result_level_num
        return true_num, reduce_one

    def static_get_one_num_mode2(self, now_num, ratio, result_level_num):
        if result_level_num > (result_level_num * ratio) + now_num:
            true_num = result_level_num * ratio + now_num
            reduce_one = 0
        else:
            true_num = result_level_num
            reduce_one = result_level_num * ratio + now_num - result_level_num
        return true_num, reduce_one

    def static_get_one_num_mode1(self, now_num, ratio, result_level_num):
        if result_level_num < (result_level_num + now_num) * ratio:
            true_num = result_level_num
            reduce_one = (result_level_num + now_num) * ratio - result_level_num
        else:
            true_num = (result_level_num + now_num) * ratio
            reduce_one = 0
        return true_num, reduce_one

    def static_get_one_num_mode0(self, now_num, result_level_num):
        true_num = result_level_num
        reduce_one = now_num
        return true_num, reduce_one

    def static_get_zero_index(self, zero_num_result, zero_level_result, zero_level, mode=0, reverse=None):
        index_result = []
        num_result = []
        for i in range(len(zero_level)):
            if mode == 0:
                index_list, num = self.static_zero_random_list(zero_level_result[i], zero_num_result[i][1])
            elif mode == 1:
                index_list, num = self.static_zero_follow_mark(zero_level_result[i], zero_num_result[i][1], reverse)
            else:
                raise ModeError(str(mode) + "in static_get_zero_index")
            index_result.append(index_list)
            num_result.append(num)
        return index_result, num_result

    def static_get_zero(self, marked_zero_area_location, result_level, zero_num, config: Config):
        zero_num_result, zero_result_reduce, zero_level_result, zero_num_level, zero_level = self.static_zero_list_num(
            marked_zero_area_location, result_level, zero_num, config.zero_ratio, config.zero_num_mode)
        index_result, num_result = self.static_get_zero_index(zero_num_result, zero_level_result, zero_level,
                                                              config.get_zero_index_mode, reverse=False)
        return zero_level_result, zero_result_reduce, index_result, num_result

    def static_get_one(self, marked_area_location, result_level, one_num, config: Config):
        zero_num_result, zero_result_reduce, zero_level_result, zero_num_level, zero_level = self.static_zero_list_num(
            marked_area_location, result_level, one_num, config.zero_ratio, config.zero_num_mode)
        index_result, num_result = self.static_get_zero_index(zero_num_result, zero_level_result, zero_level,
                                                              config.get_zero_index_mode, reverse=False)
        return zero_level_result, zero_result_reduce, index_result, num_result

    def process_single(self, name, path, point_array, level_array, level, level_img, patch_size, image_label, max_num,
                       zero_num, one_num, config: Config):
        # 处理单张切片的样例流程，config引入对参数的设定
        if config.use_level_array is True:
            level = None  # 如果使用level_array，将level置None，以免使用
        elif config.use_level_array is False:
            level_array = None  # 如果不使用level_array，将level_array置None，以免使用
        else:
            raise ExistError("use_level_array" + str(config.use_level_array))
        slide = self.read_slide(path)
        level_downsamples = slide.level_downsamples
        level_dimensions = slide.level_dimensions
        img, level_img = self.read_image(slide, level_img)  # 读入图像
        img = self.static_thresh_patch(img)  # 二值化图像
        if config.use_start_point is True:  # 查看是否使用start_point
            start_point = config.start_point
        else:
            start_point = (0, 0)
        if config.use_area_size is True:  # 查看是否使用area_size
            area_size = config.area_size
        else:
            area_size = None
        if config.use_start_point is True or config.use_area_size is True:
            detect_location, area_location = self.static_calculate_particular_area_point(point_array, patch_size,
                                                                                         level_downsamples,
                                                                                         start_point,
                                                                                         area_size, level_array,
                                                                                         level)
            # 将注视点矩阵转换成代表patch的detect_location, area_location
        else:
            detect_location, area_location = self.static_calculate_area_point(point_array, patch_size,
                                                                              level_downsamples, level_array, level)
        background_result = self.detect_background(img, area_location, level_img, level_downsamples)
        # 检测是否有在背景上的patch
        area_location = self.static_del_list_position_bool(area_location, background_result)
        detect_location = self.static_del_list_position_bool(detect_location, background_result)
        # 删除在背景上的patch
        area_location, detect_location = self.detect_point_num(area_location, detect_location, config.threshold)
        # 根据阈值删除注视点数量不达标的patch
        if config.detect_edge is True:
            edge_result = self.detect_edge(area_location, None, None, patch_size, level_downsamples, level_dimensions,
                                           start_point, area_size)
            # 检测是否有超出边缘的点，如有指定area_size，config.detect_edge应设为True
            area_location = self.static_del_list_position_bool(area_location, edge_result)
            detect_location = self.static_del_list_position_bool(detect_location, edge_result)
        detect_level, result_level = self.static_gain_level_positive(area_location, 2)
        # 获取area_location使用的level，及各level下数量
        # all_list = self.static_all_list(level_dimensions, detect_level, level_downsamples, patch_size, 0, 2)
        # 基于detect_level获得包含的level的所有patch
        # all_area_location, all_detect_location = self.static_separate_all_list_location_type_2(all_list)
        # 已使用新的解决方案
        all_area_location = self.static_all_list(level_dimensions, detect_level, level_downsamples, patch_size, 0, 0)
        all_detect_location = self.static_area_location2detect_location(all_area_location)
        background_result = self.detect_background(img, all_area_location, level_img, level_downsamples)
        all_area_location = self.static_del_list_position_bool(all_area_location, background_result)
        all_detect_location = self.static_del_list_position_bool(all_detect_location, background_result)
        if config.detect_edge is True:
            edge_result = self.detect_edge(all_area_location, None, None, patch_size, level_downsamples,
                                           level_dimensions, start_point, area_size)
            all_area_location = self.static_del_list_position_bool(all_area_location, edge_result)
            all_detect_location = self.static_del_list_position_bool(all_detect_location, edge_result)
        all_area_center_location = self.static_start2center_list(all_area_location, level_downsamples)
        # 将储存的patch左上角坐标转换成patch中心位置坐标
        all_distance = self.static_all_distance(all_area_center_location, location_type=(0, 1),
                                                mode=config.distance_mode)
        # 计算所有patch的距离
        all_area_location = self.static_all_list_point_num_repair(all_area_location, area_location)
        # 将all_area_location中的注视点数量部分根据area_location换为真实数量（之前都是0）
        all_area_location_mark = self.static_transform_num2mark(all_area_location)
        # 将注视点数量转换成mark
        all_mark = self.static_calculate_point_group_mark(all_area_location_mark, config.group_mark_mode, False)
        # 计算所有patch之间的权重（基于mark）
        all_mark_distance = self.static_calculate_point_group_distance_mark(all_mark, all_distance, False)
        # 根据权重和距离，计算所有patch之间的连接的重要性
        all_point_mark = self.static_calculate_all_point_get_mark(all_mark_distance, config.point_mark_mode)
        # 根据所有patch之间的连接的重要性计算每个patch的重要性
        zero_result = self.static_detect_exist_point_array(all_area_location, area_location, 1, 1)
        # 对all_area_location检测哪些属于1，那些属于0
        zero_area_location_mark, area_location_mark = self.static_separate_list_position_bool(all_point_mark,
                                                                                              zero_result)
        # 对patch的重要性的矩阵拆分出分别属于1和0
        zero_area_location, another_area_location = self.static_separate_list_position_bool(all_area_location,
                                                                                            zero_result)
        marked_zero_area_location = self.static_create_negative_marked_area_location(zero_area_location_mark,
                                                                                     zero_area_location)
        marked_area_location = self.static_create_positive_marked_area_location(area_location_mark,
                                                                                another_area_location, area_location,
                                                                                (0, 0))
        zero_level_result, zero_result_reduce, zero_index_result, zero_num_result = self.static_get_zero(
            marked_zero_area_location, result_level, zero_num, config)
        one_level_result, one_result_reduce, one_index_result, one_num_result = self.static_get_one(
            marked_area_location, result_level, one_num, config)
        single_result = (
            one_level_result, one_result_reduce, one_index_result, one_num_result, zero_level_result,
            zero_result_reduce, zero_index_result, zero_num_result)
        return single_result

    def static_calculate_num(self, reduce_num, mode=0):
        if mode == 0:
            num = reduce_num
        else:
            raise ModeError(str(mode) + "in static_calculate_num")
        return num

    def static_sum_list_num(self, list_a):
        all_item = []
        def list_num(list_b):
            for x in list_b:
                if type(x) is not list:
                    all_item.append(x)
                else:
                    list_num(x)
            return sum(all_item)
        return list_num(list_a)

    def process(self, name_array=None, path=None, point_array=None, level_array=None, level=None, image_label=None,
                max_num=None, config: Config = None):
        zero_num = None
        one_num = None
        result_train = []
        result_val = []
        train_total_one_num = 0
        train_total_zero_num = 0
        val_total_one_num = 0
        val_total_zero_num = 0
        use_list = self.static_random_class_index(config.class_ratio, len(name_array))
        for i in range(len(name_array)):
            if i in use_list[0]:
                single_result = self.process_single(name_array[i], path[i], point_array[i], level_array[i], config.level,
                                                    config.level_img, config.patch_size, image_label[i], max_num[i],
                                                    zero_num, one_num, config)
                result_train.append(single_result)
                train_total_one_num += self.static_sum_list_num(single_result[3])
                train_total_zero_num += self.static_sum_list_num(single_result[3])
                one_num = self.static_calculate_num(single_result[1], config.calculate_one_num_mode)
                zero_num = self.static_calculate_num(single_result[5], config.calculate_zero_num_mode)
            # elif i in use_list[1]:
            #     single_result = self.process_single(name_array[i], path[i], point_array[i], level_array[i],
            #                                         config.level,
            #                                         config.level_img, config.patch_size, image_label[i], max_num[i],
            #                                         zero_num, one_num, config)
            #     result_train.append(single_result)
            #     train_total_one_num += self.static_sum_list_num(single_result[3])
            #     train_total_zero_num += self.static_sum_list_num(single_result[3])
            #     one_num = self.static_calculate_num(single_result[1], config.calculate_one_num_mode)
            #     zero_num = self.static_calculate_num(single_result[5], config.calculate_zero_num_mode)
            elif i in use_list[2]:
                pass
            else:
                raise OutBoundError(str(i) + "in process")
        information = {'name': name_array, 'path': path, 'point': point_array, 'level': level_array,
                       'label': image_label, 'max_num': max_num}
        return information, result, total_one_num, total_zero_num

    def read

    def static_save_information_key(self, information, key, output_direc=None):
        value = np.array(information[key])
        if output_direc is not None:
            path_name = os.path.join(output_direc, key + ".npy")
        else:
            path_name = key + ".npy"
        np.save(path_name, value)

    def static_save_information(self, information, output_direc=None):
        for i in list(information.keys()):
            self.static_save_information_key(information, i, output_direc)

    def static_save_result(self, result, output_direc):
        result = np.array(result)
        if output_direc is not None:
            path_name = os.path.join(output_direc, "patch" + ".npy")
        else:
            path_name = "patch" + ".npy"
        np.save(path_name, result)

    def static_read_save_patch(self):


    def save(self, information: dict=None, result=None, output_direc=None, mode=0):
        if mode == 0:
            self.static_save_information(information, output_direc)
            self.static_save_result(result, output_direc)
        elif mode == 1:
            self.static_save_information(information, output_direc)
            self.static_save_result(result, output_direc)


