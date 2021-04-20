from recReader import *
from math import *
from copy import *
import numpy as np
import os


class DataProcesser:
    def __init__(self, data, ppm):
        self.__av = []
        self.__data = deepcopy(data)
        self.mid_value(data, 2)
        self.calc_angular_v(self.__data, ppm)

    def mid_value(self, data, m: int):
        np_data = np.array(data)
        for i in range(m, len(data) - m):
            self.__data[i][0] = np.median(np_data[i - m: i + m + 1, 0: 1])
            self.__data[i][1] = np.median(np_data[i - m: i + m + 1, 1: 2])

    def calc_angular_v(self, data, ppm):
        for i in range(len(data) - 1):
            if data[i][3] != data[i + 1][3]:  # 这一帧是倍率切换帧，不处理，用上一帧数据代替
                self.__av.append(self.__av[-1])
                continue
            this_data = data[i]
            this_level = this_data[3]
            next_data = data[i + 1]
            next_level = next_data[3]
            if isnan(this_data[2]):  # 表示此帧眼动仪未取到头部位置
                if isnan(next_data[2]):  # 下一帧为nan
                    self.__av.append(0)
                    continue
                else:
                    h = next_data[2]
            else:
                if isnan(next_data[2]):  # 下一帧为nan
                    h = this_data[2]
                else:
                    h = (this_data[2] + next_data[2]) / 2  # 取平均距离
            sx = (this_data[5] + next_data[5]) / 2 / this_level  # 求屏幕中点平均坐标
            sy = (this_data[6] + next_data[6]) / 2 / next_level
            ac = dis(this_data[0: 2], [sx, sy]) / this_level / ppm
            ad = square_root(ac, h)
            bc = dis(next_data[0: 2], [sx, sy]) / next_level / ppm
            bd = square_root(bc, h)
            ab = dis([this_data[0] / this_level, this_data[1] / this_level],
                     [next_data[0] / next_level, next_data[1] / next_level]) / ppm
            r = (pow(ad, 2) + pow(bd, 2) - pow(ab, 2)) / (2 * ad * bd)
            if r > 1:
                r = 1
            elif r < 0:
                r = 0
            angle = acos(r)
            rad = angle / (next_data[-1] - this_data[-1])
            degree = 180 * rad
            self.__av.append(degree)

    def get_data(self):
        return self.__data

    def get_angular_v(self):
        return self.__av


def dis(p1, p2):
    return square_root(p1[0] - p2[0], p1[1] - p2[1])


def square_root(a, b):
    """
        求算数平方根
    """
    return sqrt(pow(a, 2) + pow(b, 2))


def number_type(typeis):
    if typeis == "黑色素瘤":
        real_type = 1
    elif typeis == "其他":
        real_type = 0
    elif typeis == "基底细胞癌":
        real_type = 1
    elif typeis == "良性痣":
        real_type = 0
    else:
        print("error,type is ", typeis)
        real_type = None
    return real_type


def number_type4(typeis):
    if typeis == "黑色素瘤":
        real_type = 1
    elif typeis == "其他":
        real_type = 0
    elif typeis == "基底细胞癌":
        real_type = 2
    elif typeis == "良性痣":
        real_type = 3
    else:
        print("error,type is ", typeis)
        real_type = None
    return real_type


def all_reader(directory_name):
    array_of_name = []
    namelist = []
    array_of_img = []
    t_num = []
    slide_name = []
    type_all = []
    array_of_level = []
    for filename in os.listdir(directory_name):
        file_name_clean = filename[:-4]
        if file_name_clean not in array_of_name:
            array_of_name.append(file_name_clean)
            erc_name = directory_name + "/" + file_name_clean + '.erc'
            rec_name = directory_name + "/" + file_name_clean + '.rec'
            print(file_name_clean)
            try:
                data_want = []
                level_list = []
                rec = RECreader(rec_name, erc_name)
                slide_file_name = rec.get_slide_file_name()
                slide_name.append(slide_file_name)
                raw_data = rec.get_data()
                now_type = rec.get_type()
                num_type = number_type(now_type)
                type_all.append(num_type)
                screen = rec.get_screen_info()
                ppm = (screen[0] / screen[2] + screen[1] / screen[3]) / 2
                proc = DataProcesser(raw_data, ppm)
                data_list = proc.get_data()
                for i in range(len(data_list)):
                    data_want.append(data_list[i][0:2])
                    level_list.append(data_list[i][3])
                t_num.append(rec.get_tick_num())
                array_of_img.append(data_want)
                array_of_level.append(level_list)
                namelist.append(file_name_clean)
            except:
                print(r"can't read:", file_name_clean)
    return namelist, array_of_img, array_of_level, slide_name, t_num, type_all


def all_reader_type4(directory_name):
    array_of_name = []
    namelist = []
    array_of_img = []
    t_num = []
    slide_name = []
    type_all = []
    for filename in os.listdir(directory_name):
        file_name_clean = filename[:-4]
        if file_name_clean not in array_of_name:
            array_of_name.append(file_name_clean)
            erc_name = directory_name + "/" + file_name_clean + '.erc'
            rec_name = directory_name + "/" + file_name_clean + '.rec'
            print(file_name_clean)
            try:
                data_want = []
                rec = RECreader(rec_name, erc_name)
                slide_file_name = rec.get_slide_file_name()
                slide_name.append(slide_file_name)
                raw_data = rec.get_data()
                now_type = rec.get_type()
                num_type = number_type4(now_type)
                type_all.append(num_type)
                screen = rec.get_screen_info()
                ppm = (screen[0] / screen[2] + screen[1] / screen[3]) / 2
                proc = DataProcesser(raw_data, ppm)
                data_list = proc.get_data()
                for i in range(len(data_list)):
                    data_want.append(data_list[i][0:2])
                t_num.append(rec.get_tick_num())
                array_of_img.append(data_want)
                namelist.append(file_name_clean)
            except:
                print(r"can't read:", file_name_clean)
    return namelist, array_of_img, slide_name, t_num, type_all


def label_reader(directory_name):
    array_of_name = []
    type_all = []
    for filename in os.listdir(directory_name):
        file_name_clean = filename[:-4]
        if file_name_clean not in array_of_name:
            array_of_name.append(file_name_clean)
            erc_name = directory_name + "/" + file_name_clean + '.erc'
            rec_name = directory_name + "/" + file_name_clean + '.rec'
            print(file_name_clean)
            try:
                rec = RECreader(rec_name, erc_name)
                type_is = rec.get_type()
                type_all.append(type_is)
            except:
                print(r"can't read:", file_name_clean)
    return type_all


if __name__ == "__main__":  # 这里是示例用法
    print("not here")
