import tool.dataProcesser as Da
import openslide
import cv2
import numpy as np
import os
import random  # only this


# this file is based on slidelabel_dataset
# this file is going to separate the data to val, test, train
# each action which only belongs to this file will be added "only this" as note


def name2path(name, inputdirec):
    """

    :param name: save filepath
    :param inputdirec: real data filepath
    :return: file path now
    """
    chongzhuname = name.replace("\\", '/')
    chongzhuinputdirec = inputdirec.replace("\\", '/')
    [_, truename] = os.path.split(chongzhuname)
    if inputdirec[-1] == '/' or inputdirec[-1] == "\\":
        newpath = chongzhuinputdirec + truename
    else:
        newpath = chongzhuinputdirec + "/" + truename
    return newpath


def name2pathinlist(name, inputdirec):
    """

    :param name: save file path in list
    :param inputdirec:  real data filepath
    :return:  file path now
    """
    pathlist = []
    for i in range(len(name)):
        path = name2path(name[i], inputdirec)
        pathlist.append(path)
    return pathlist


erc_path = r"D:\ajmq\point_cs"
patch_size = 128
use_size = patch_size * 3
down_set = 2
auto_set = 1
background_set = 1
now_level = 3
uselevel = 1
use_background = 1
slidenum = "all"
now_mode = "same"
thresh_hold = 2
one_cat_mode = "all"
zero_cat_mode = "same"
process_direc = r"D:\ajmq\process"
input_direc = r"D:\ajmq\data_no_process"
repeat_process_mode = "add"
set_num = 4  # only this
ratio_for_class = np.array([5, 5, 1, 1])  # only this
class_of_name = np.array(["train", "train2", "val", "val2"])  # only this
class_set_mode = "total"  # or "separate"  # only this
regular = 20  # only this file


# down_level/down_set指的是当使用的patch_size过小时，选取更小的level来扩大选取区域的大小， 当前level-down_level=实际读取level
# auto_level/auto_set指的是是否自动计算down_level的值
# mode/now_mode指的是正样本与负样本的分配模式，例如“same”代表着数量一致
# slidenum/slide_num指的是样本数量，在mode为“same”情况下，“all”代表没有进一步处理
# one_cat/one_cat_mode指的是不同level下正样本的处理方式，例如“all”，代表全部都要
# zero_cat/zero_cat_mode同理，这指的是负样本，例如“same”，代表不同level下数量一致
# repeat_mode/repeat_process_mode指的是遇到重复情况时，即同一张切片被看超过或等于两次，的处理方案，例如“add”代表注视点集合并，其他代表不处理
# set_class/set_num指的是拆分数据集成几份,默认为3，即划分train、val、test
# ratio_class/ratio_for_class指的是三份的比例，不要求加起来等于1
# class_name/class_of_name指的是拆分后储存位置的文件名
# class_mode=class_set_mode指的是拆分的模式，例如“separate”是指按文件拆分，“total”是指按所有点拆分
# 样例之外的方法未编写


class GetSlide:
    def __init__(self, ercpath=erc_path, size=use_size, little_patch=patch_size, inputdirec=input_direc,
                 level=now_level, use_level=uselevel, down_level=down_set, auto_level=auto_set,
                 background_use=use_background, background_level=background_set, regular_set=regular,
                 mode=now_mode, slide_num=slidenum, one_cat=one_cat_mode, zero_cat=zero_cat_mode,
                 threshold=thresh_hold, output_direc=process_direc, repeat_mode=repeat_process_mode, set_class=set_num,
                 ratio_class=ratio_for_class, class_name=class_of_name, class_mode=class_set_mode):
        print("starting init...")
        [name, pointarray, levelarray, path, max_num, imagelabel] = Da.all_reader(ercpath)
        print("finish read point file")
        print("max num:{}".format(sum(max_num)))
        if repeat_mode == "add":
            print("use repeat_mode as add")
            [name, pointarray, levelarray, path, imagelabel] = self.repeat_add(name, pointarray, levelarray, path,
                                                                               imagelabel)
            print("finish add operation")
        self.repeat_mode = repeat_mode
        if auto_level == 1:
            enlarge = 1
            down_cal = 0
            while True:
                if little_patch * enlarge >= 224:
                    down_level = down_cal
                    break
                else:
                    enlarge = enlarge * 2
                    down_cal = down_cal + 1
            little_patch = little_patch * (2 ** down_level)
            size = size * (2 ** down_level)
        print("use down_level:", down_level)
        self.down_level = down_level
        self.background_use = background_use
        self.background_level = background_level
        self.regular = regular_set
        print(self.regular)
        print(self.regular is not None)
        self.set_class = set_class
        self.ratio_class = ratio_class
        self.class_name = class_name
        self.class_mode = class_mode
        self.namearray = name
        self.pointarray = pointarray
        self.imagelabel = imagelabel
        path = name2pathinlist(path, inputdirec)
        print("finish catch image path")
        self.patharray = path
        self.ercpath = ercpath
        self.patchsize = size
        self.little_patch_size = little_patch
        self.inputdirec = inputdirec
        self.level = level
        self.levelarray = levelarray
        self.use_level = use_level
        self.mode = mode
        self.num = slide_num
        self.threshold = threshold
        self.one_cat = one_cat
        self.zero_cat = zero_cat
        self.slide_total = 0
        self.output_direc = output_direc
        self.re_level()
        if self.class_mode == "separate" or self.class_mode == "total":
            self.ratio_class_to_1 = []
        # only this
        print("starting set_data")
        self.set_data()

    def re_level(self):
        for v in range(len(self.namearray)):
            for w in range(len(self.levelarray[v])):
                div = 1
                con = 0
                while True:
                    if self.levelarray[v][w] == div:
                        self.levelarray[v][w] = con
                        break
                    con = con + 1
                    div = div * 2

    def repeat_add(self, name_list, point_list, level_list, path_list, image_label):
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
        return name_list, point_list, level_list, path_list, image_label

    def detectbackground(self, img, x, y):
        try:
            if img[y, x] == 255:
                background = 1
            else:
                background = 0
        except:
            background = 1
        return background

    def detectoutside(self, area_location, threshold):
        after_processing = []
        for i in area_location:
            if i[0] >= threshold:
                after_processing.append(i)
        return after_processing

    def detectpointarray(self, point_array, the_patch_size, level_array, level_downsamples, use_level=1, down_level=0):
        if use_level == 0:
            level = level_array
            area_location = []
            detect_location = []
            if level - down_level >= 0:
                the_patch_size = the_patch_size // (2 ** down_level)
            else:
                the_patch_size = the_patch_size // (2 ** level)
            level_point_array = self.mul_div(
                self.mul_div(point_array, level_downsamples[0], level_downsamples[level], 1), div=the_patch_size,
                as_int=1)
            for j in range(len(point_array)):
                [x, y] = self.mul_div(point_array[j], level_downsamples[0], level_downsamples[level], 1)
                id_x = int(x / the_patch_size)
                id_y = int(y / the_patch_size)
                if [id_x, id_y, level] not in detect_location:
                    number_point = self.detectpointexist(level_point_array, id_x, id_y)
                    detect_location.append([id_x, id_y, level])
                    if level - down_level >= 0:
                        level_new = level - down_level
                    else:
                        level_new = 0
                    area_location.append([number_point, id_x, id_y, level_new])
        else:
            area_location = []
            detect_location = []
            for j in range(len(point_array)):
                level = level_array[j]
                level = int(level)
                if level - down_level >= 0:
                    the_patch_size_now = the_patch_size // (2 ** down_level)
                else:
                    the_patch_size_now = the_patch_size // (2 ** level)
                [x, y] = self.mul_div(point_array[j], level_downsamples[0], level_downsamples[level], 1)
                id_x = int(x / the_patch_size_now)
                id_y = int(y / the_patch_size_now)
                level_point_array = self.mul_div(
                    self.mul_div(point_array, level_downsamples[0], level_downsamples[level], 1),
                    div=the_patch_size_now,
                    as_int=1)
                if [id_x, id_y, level] not in detect_location:
                    number_point = self.detectpointexist(level_point_array, id_x, id_y)
                    detect_location.append([id_x, id_y, level])
                    if level - down_level >= 0:
                        level = level - down_level
                    else:
                        level = 0
                    area_location.append([number_point, id_x, id_y, level])
        return area_location

    def mul_div(self, a_list, mul=None, div=None, as_int=None):
        a_list = np.array(a_list)
        if mul is not None:
            a_list = a_list * mul
        if div is not None:
            a_list = a_list / div
        if as_int == 1:
            a_list = a_list.astype(int)
        a_list = a_list.tolist()
        return a_list

    def detectpointexist(self, point_array, x, y):
        number_point = point_array.count([x, y])
        return number_point

    def readimage(self, idx, level=None):
        if level is not None:
            level = level
        elif self.use_level == 0:
            level = self.level + 2
        else:
            level = 3
        path = self.patharray[idx]
        print(path)
        slide = openslide.OpenSlide(path)
        [lengthofimg, heightofimg] = slide.level_dimensions[level]
        img = np.array(slide.read_region([0, 0], level, [lengthofimg, heightofimg]))[:, :, :3]
        return img, level

    def readimagearea(self, idx, idy, slide, level):
        x = idx * self.patchsize
        y = idy * self.patchsize
        level_downsamples = slide.level_downsamples
        x = int(x * level_downsamples[level] / level_downsamples[0])
        y = int(y * level_downsamples[level] / level_downsamples[0])
        img = np.array(slide.read_region([x, y], level, [self.patchsize, self.patchsize]))[:, :, :3]
        return img

    def set_label(self, idx, idy, level, area_location):
        detect_location = [i[1:] for i in area_location]
        if [idx, idy, level] not in detect_location:
            label = 0
        else:
            label = 1
        return label

    def detect_edge(self, idx, idy, slide, level):
        [lengthofimg, heightofimg] = slide.level_dimensions[level]
        length = int((lengthofimg - 1) / self.patchsize)
        height = int((heightofimg - 1) / self.patchsize)
        if idx == length:
            x_edge = 1
            if (length + 1) * self.patchsize == lengthofimg:
                x_out = 0
            else:
                x_out = 1
        else:
            x_edge = 0
            x_out = 0
        if idy == height:
            y_edge = 1
            if (height + 1) * self.patchsize == heightofimg:
                y_out = 0
            else:
                y_out = 1
        else:
            y_edge = 0
            y_out = 0
        return x_edge, x_out, y_edge, y_out

    def thresh(self, img):
        [x, y, _] = np.shape(img)
        w = int(x // self.patchsize)
        h = int(y // self.patchsize)
        img = img[0:(w * self.patchsize), 0:(h * self.patchsize), :]
        img = cv2.resize(img, (h, w), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret2, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img

    def change_pixel(self, ida, size1, size2):
        id_min = int(ida * size1 / size2)
        if id_min < (ida * size1 / size2):
            id_min = id_min + 1
        id_max = int((ida + 1) * size1 / size2)
        if id_max < ((ida + 1) * size1 / size2):
            id_max = id_max + 1
        return list(range(id_min, id_max))

    def read_save(self, read_list, point_with_path, class_num):
        for t in read_list:
            [name, path, index_x, index_y, point_level, label_list] = point_with_path[t]
            slide = openslide.OpenSlide(path)
            area_img = self.readimagearea(index_x, index_y, slide, point_level)
            np.save(self.output_direc + self.class_name[class_num] + "/{}-{}-{}-{}.npy".format(name, index_x, index_y,
                                                                                               point_level),
                    np.array(label_list))
            cv2.imwrite(
                self.output_direc + self.class_name[class_num] + "/{}-{}-{}-{}.jpg".format(name, index_x, index_y,
                                                                                           point_level),
                area_img)

    def check_zero(self, img, level, area_location, level_downsamples):
        [img_x, img_y] = np.shape(img)
        zero_point_array = []
        zero_point_level_array = []
        zero_total = 0
        zero_level_total = []
        print("start approach false samples")
        for m in range(level + 1):
            zero_point_level = []
            zero_level_num = 0
            for zero_x in range(img_x):
                for zero_y in range(img_y):
                    if img[zero_x, zero_y] == 0:
                        [zero_list, zero_num] = self.get_zero(zero_x, zero_y, level, m, area_location,
                                                              level_downsamples)
                        zero_point_level = zero_point_level + zero_list
                        zero_point_array = zero_point_array + zero_list
                        zero_level_num = zero_level_num + zero_num
            zero_point_level_array.append(zero_point_level)
            zero_total = zero_total + zero_level_num
            zero_level_total.append(zero_level_num)
        return zero_point_array, zero_point_level_array, zero_total, zero_level_total

    def get_zero(self, zero_x, zero_y, level, test_level, area_location, level_downsamples):
        zero_min_x = int(zero_x * level_downsamples[level] / level_downsamples[test_level])
        zero_min_y = int(zero_y * level_downsamples[level] / level_downsamples[test_level])
        zero_max_x = int((zero_x + 1) * level_downsamples[level] / level_downsamples[test_level])
        zero_max_y = int((zero_y + 1) * level_downsamples[level] / level_downsamples[test_level])
        x_list = list(range(zero_min_x, zero_max_x))
        y_list = list(range(zero_min_y, zero_max_y))
        zero_list = []
        zero_num = 0
        for x in x_list:
            for y in y_list:
                if self.set_label(x, y, test_level, area_location) == 0:
                    zero_list.append([x, y, test_level])
                    zero_num = zero_num + 1
        return zero_list, zero_num

    def check_level(self, area_location):
        level_list = []
        level_num = []
        detect_level = [i[3] for i in area_location]
        for area_simple in area_location:
            level = area_simple[3]
            if level not in level_list:
                level_list.append(level)
                level_count = detect_level.count(level)
                level_num.append([level, level_count])
        return level_num

    def max_level(self, area_location):
        detect_level = [i[3] for i in area_location]
        max_l = max(detect_level)
        return max_l

    def get_location(self, level_list, level):
        detect_level = [i[0] for i in level_list]
        try:
            index_l = detect_level.index(level)
        except:
            index_l = None
        return index_l

    def set_data(self):
        len_slide = len(self.namearray)
        if self.class_mode == "total":
            point_with_path = []
            path_little = []
        elif self.class_mode == "separate":
            random_list = list(range(len_slide))
            random.shuffle(random_list)
            num_class_list = []
            len_all = 0
            for r in range(self.set_class):
                self.ratio_class_to_1.append(self.ratio_class[r] / sum(self.ratio_class))
                if r + 1 < self.set_class:
                    len_class = int(self.ratio_class[r] * len_slide / sum(self.ratio_class))
                    num_class_list.append(random_list[len_all:(len_all + len_class)])
                    len_all = len_all + len_class
                else:
                    num_class_list.append(random_list[len_all:len_slide])
        print("slide num:", len_slide)
        for k in range(len_slide):
            print("now slide:", k, "/", len_slide)
            print("start reading image to detect background")
            img, level = self.readimage(k, self.background_level)
            print("start separating background and image")
            img = self.thresh(img)
            slide = openslide.OpenSlide(self.patharray[k])
            level_downsamples = slide.level_downsamples
            print("start detecting the area where points are in")
            area_location = self.detectpointarray(self.pointarray[k], self.patchsize, self.levelarray[k],
                                                  level_downsamples, self.use_level, self.down_level)
            little_area_location = self.detectpointarray(self.pointarray[k], self.little_patch_size, self.levelarray[k],
                                                         level_downsamples, self.use_level, self.down_level)
            print("detect if points are outside")
            area_location = self.detectoutside(area_location, self.threshold)
            area_number = 0
            area_test = 0
            real_area_location = []
            one_point_array = []

            print("start processing each point")
            print("len:", len(area_location))

            for m in range(len(area_location)):
                [idx, idy, point_level] = area_location[m][1:]
                level_x = int(idx * level_downsamples[point_level] / level_downsamples[level])
                level_y = int(idy * level_downsamples[point_level] / level_downsamples[level])
                if self.background_use == 1:
                    background = self.detectbackground(img, level_x, level_y)
                    if background == 0:
                        [_, x_out, _, y_out] = self.detect_edge(idx, idy, slide, area_location[m][3])
                        area_test = area_test + 1
                        if x_out == 0 and y_out == 0:
                            real_area_location.append([area_location[m][0], idx, idy, area_location[m][3]])
                            one_point_array.append([idx, idy, area_location[m][3]])
                            area_number = area_number + 1
                else:
                    [_, x_out, _, y_out] = self.detect_edge(idx, idy, slide, area_location[m][3])
                    area_test = area_test + 1
                    if x_out == 0 and y_out == 0:
                        real_area_location.append([area_location[m][0], idx, idy, area_location[m][3]])
                        one_point_array.append([idx, idy, area_location[m][3]])
                        area_number = area_number + 1

            if area_number == 0:
                if self.background_use == 1:
                    img, level = self.readimage(k, 0)
                    img = self.thresh(img)
                    area_number = 0
                    area_test = 0
                    real_area_location = []
                    one_point_array = []
                    for m in range(len(area_location)):
                        [idx, idy, point_level] = area_location[m][1:]
                        level_x = int(idx * level_downsamples[point_level] / level_downsamples[level])
                        level_y = int(idy * level_downsamples[point_level] / level_downsamples[level])
                        if self.detectbackground(img, level_x, level_y) == 0:
                            [_, x_out, _, y_out] = self.detect_edge(idx, idy, slide, area_location[m][3])
                            area_test = area_test + 1
                            if x_out == 0 and y_out == 0:
                                real_area_location.append([area_location[m][0], idx, idy, area_location[m][3]])
                                one_point_array.append([idx, idy, area_location[m][3]])
                                area_number = area_number + 1
            if area_number == 0:
                real_area_location = area_location
                area_number = len(area_location)
                one_point_array = [i[1:] for i in area_location]
            print(area_test)
            print("finish processing each point")
            print("final len:", area_number)
            if self.mode == "same":
                zero_number = area_number
            else:
                zero_number = 0
                print("maybe something wrong")

            max_l = self.max_level(real_area_location)
            [zero_point_array, zero_point_level_array, zero_total, zero_level_total] = self.check_zero(img, max_l,
                                                                                                       area_location,
                                                                                                       level_downsamples)
            print(zero_number)
            if self.mode == "same":
                if self.zero_cat == "same":
                    level_now = 0
                    level_num = self.check_level(real_area_location)
                    zero_point_array_new = []
                    for zero_now_array in zero_point_level_array:
                        index_l = self.get_location(level_num, level_now)
                        if index_l is not None:
                            zero_level_now = level_num[index_l][1]
                        else:
                            zero_level_now = 0
                        len_array = len(zero_now_array)
                        if zero_level_now < len_array:
                            len_zero = zero_level_now
                        else:
                            len_zero = len_array
                        random_zero = list(range(len_array))
                        random.shuffle(random_zero)
                        real_read = random_zero[:len_zero]
                        for index in real_read:
                            zero_point_array_new.append(zero_now_array[index])

            if self.mode != "same":
                if self.zero_cat == "same":
                    level_now = 0
                    level_num = self.check_level(real_area_location)
                    zero_point_array_new = []
                    for zero_now_array in zero_point_level_array:
                        index_l = self.get_location(level_num, level_now)
                        level_now = level_now + 1
                        if index_l is not None:
                            zero_level_now = level_num[index_l][1]
                            if zero_level_now * self.mode < 1:
                                zero_level_now = 1
                            else:
                                zero_level_now = int(zero_level_now * self.mode)
                        else:
                            zero_level_now = 0
                        len_array = len(zero_now_array)
                        if zero_level_now < len_array:
                            len_zero = zero_level_now
                        else:
                            len_zero = len_array
                        random_zero = list(range(len_array))
                        random.shuffle(random_zero)
                        real_read = random_zero[:len_zero]
                        for index in real_read:
                            zero_point_array_new.append(zero_now_array[index])

            total_point_array = one_point_array + zero_point_array_new
            if self.set_class == 1 and self.class_name is None:
                for n in range(len(total_point_array)):
                    [point_x, point_y, the_level] = total_point_array[n]
                    area_img = self.readimagearea(point_x, point_y, slide, the_level)
                    x_list = self.change_pixel(point_x, self.patchsize, self.little_patch_size)
                    y_list = self.change_pixel(point_y, self.patchsize, self.little_patch_size)
                    label_list = []
                    for p in range(len(x_list)):
                        for q in range(len(y_list)):
                            xy_label = self.set_label(x_list[p], y_list[q], the_level, little_area_location)
                            label_list.append([x_list[p], y_list[q], xy_label])
                    np.save(
                        self.output_direc + "{}-{}-{}-{}.npy".format(self.namearray[k], point_x, point_y, point_level),
                        np.array(label_list))
                    cv2.imwrite(
                        self.output_direc + "{}-{}-{}-{}.jpg".format(self.namearray[k], point_x, point_y, point_level),
                        area_img)
            elif self.class_mode == "total":
                for n in range(len(total_point_array)):
                    [point_x, point_y, the_level] = total_point_array[n]
                    area_img = self.readimagearea(point_x, point_y, slide, the_level)
                    x_list = self.change_pixel(point_x, self.patchsize, self.little_patch_size)
                    y_list = self.change_pixel(point_y, self.patchsize, self.little_patch_size)
                    label_list = []
                    for p in range(len(x_list)):
                        for q in range(len(y_list)):
                            xy_label = self.set_label(x_list[p], y_list[q], the_level, little_area_location)
                            label_list.append([x_list[p], y_list[q], xy_label])
                    point_with_path.append(
                        [self.namearray[k], self.patharray[k], point_x, point_y, the_level, label_list])
            elif self.class_mode == "separate":
                print("start saving")
                for s in range(self.set_class):
                    if k in num_class_list[s]:
                        for n in range(len(total_point_array)):
                            [point_x, point_y, the_level] = total_point_array[n]
                            area_img = self.readimagearea(point_x, point_y, slide, the_level)
                            x_list = self.change_pixel(point_x, self.patchsize, self.little_patch_size)
                            y_list = self.change_pixel(point_y, self.patchsize, self.little_patch_size)
                            label_list = []
                            for p in range(len(x_list)):
                                for q in range(len(y_list)):
                                    xy_label = self.set_label(x_list[p], y_list[q], the_level, little_area_location)
                                    label_list.append([x_list[p], y_list[q], xy_label])
                            np.save(
                                self.output_direc + self.class_name[s] + "/{}-{}-{}-{}.npy".format(self.namearray[k],
                                                                                                   point_x,
                                                                                                   point_y,
                                                                                                   point_level),
                                np.array(label_list))
                            cv2.imwrite(
                                self.output_direc + self.class_name[s] + "/{}-{}-{}-{}.jpg".format(self.namearray[k],
                                                                                                   point_x,
                                                                                                   point_y,
                                                                                                   point_level),
                                area_img)
            print("finish slide processing")
        if self.class_mode == "total":
            print("start saving")
            len_point = len(point_with_path)
            random_list = list(range(len_point))
            random.shuffle(random_list)
            len_all = 0
            for r in range(self.set_class):
                self.ratio_class_to_1.append(self.ratio_class[r] / sum(self.ratio_class))
                len_class = int(self.ratio_class[r] * len_point / sum(self.ratio_class))
                if self.regular is not None:
                    len_class = int((len_class // self.regular) * self.regular)
                read_list = random_list[len_all:(len_all + len_class)]
                print(len(read_list))
                self.read_save(read_list, point_with_path, r)
                len_all = len_all + len_class
        print("finish whole pre-processing")


if __name__ == "__main__":
    a = GetSlide()
