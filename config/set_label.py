import os
import sys

o_path = os.getcwd()
sys.path.append(o_path)
from tool.Error import TypeError


class Control:
    def __init__(self, point_is_array=True):
        self.detect_location = DetectLocation()
        self.point = Point(point_is_array)


class DetectLocation:
    def __init__(self):
        self.keep_type = False  # 不改变detect_location，直接返回
        self.index_range = [0, None]  # 取的location的index范围, 如果想依靠location_type获得，建议填写[0, None]
        self.is_level = True  # 是否是按level储存
        self.location_type = None  # 如果不清楚index_range, 在此项注明类型，会寻找匹配的index_range
        self.keep_level = True  # 是否保留level,该项在指定了location_type才有意义
        self.keep_patch_size = True  # 是否保留patch_size,该项在指定了location_type才有意义
        self.optional_operate = False  # 该项为False时，仅考虑将location，转为detect_location，不会有额外的separate操作
        self.separate_point_num = False  # 是否将注视点数量单独返回
        self.exist_point_num = True  # 该location是否包括注视点数量
        self.separate_mark = False  # 是否将mark数量单独返回
        self.exist_mark = True  # 该location是否包括mark

        self.calculate_check = False
        self.get_index_range()  # 计算匹配的index_range
        self.check_keep_type()  # 检查是否满足keep_type=1的条件
        self.check_optional_operate()  # 检查是否满足optional_operate=False的条件

    def check_optional_operate(self):
        if self.exist_mark is True and self.exist_point_num is True:
            if self.separate_point_num is True or self.separate_mark is True:
                pass
            else:
                self.optional_operate = False
        else:
            self.optional_operate = False

    def get_index_range(self):
        if self.location_type is not None:
            if self.location_type == "detect" or self.location_type == "detect_location":
                self.index_range[0] = 0
                self.get_index_range_1()
            elif self.location_type == "area" or self.location_type == "area_location":
                self.index_range[0] = 1
                self.get_index_range_1()
            elif self.location_type == "marked_area" or self.location_type == "marked_area_location":
                self.index_range[0] = 2
                self.get_index_range_1()
            elif self.location_type == "marked_detect" or self.location_type == "marked_detect":
                self.index_range[0] = 1
                self.get_index_range_1()
            else:
                raise TypeError(self.location_type + "in get_index_range")

    def get_index_range_1(self):
        if self.keep_level is True:
            if self.keep_patch_size is True:
                self.index_range[1] = None
            else:
                self.index_range[1] = -1
        else:
            self.index_range[1] = -2

    def check_keep_type(self):
        if self.index_range[0] == 0 or self.index_range[0] is None:
            if self.index_range[1] is None:
                if self.is_level is False:
                    self.keep_type = True

    def other2detect(self, detect_location):
        detect_location = detect_location.copy()
        if self.keep_type is True:
            return detect_location
        else:
            if self.is_level is True:
                detect_location_without_level = []
                for i in range(len(detect_location)):
                    detect_location_without_level += detect_location[i]
                detect_location = detect_location_without_level
            detect_location = [i[self.index_range[0]: self.index_range[1]] for i in detect_location]
            return detect_location

    def process_optional_operate(self, detect_location):
        if self.optional_operate is not False:
            mark_array = None
            point_num_array = None
            if self.exist_mark is True and self.separate_mark is True:
                mark_array = self.from_location_separate_mark(detect_location)
            if self.exist_point_num is True and self.separate_point_num is True:
                point_num_array = self.from_location_separate_point_num(detect_location)
            if mark_array is not None and point_num_array is not None:
                return mark_array, point_num_array
            elif mark_array is not None and point_num_array is None:
                return mark_array
            elif mark_array is None and point_num_array is not None:
                return point_num_array
            else:
                return None
        else:
            return None

    @staticmethod
    def from_location_separate_mark(detect_location):
        detect_location = detect_location.copy()
        mark_array = [i[0] for i in detect_location]
        return mark_array

    def from_location_separate_point_num(self, detect_location):
        detect_location = detect_location.copy()
        if self.exist_point_num is True:
            point_num_array = [i[1] for i in detect_location]
        else:
            point_num_array = [i[0] for i in detect_location]
        return point_num_array

    def transform_detect_location(self, detect_location):
        self.get_index_range()
        separate_result = self.process_optional_operate(detect_location)
        detect_location = self.other2detect(detect_location)
        if separate_result is not None:
            return detect_location, separate_result
        else:
            return detect_location

    def get_index_and_check(self):
        if self.calculate_check is False:
            self.get_index_range()  # 计算匹配的index_range
            self.check_keep_type()  # 检查是否满足keep_type=1的条件
            self.check_optional_operate()  # 检查是否满足optional_operate=False的条件
            self.calculate_check = True


class Point:
    def __init__(self, is_array=True):
        self.keep_type = False  # 不改变detect_location，直接返回
        self.index_range = [0, None]  # 取的location的index范围, 如果想依靠location_type获得，建议填写[0, None]
        self.is_array = is_array
        self.is_array_init = is_array
        self.is_level = True  # 是否是按level储存, 该项仅在is_array=True有效
        self.location_type = None  # 如果不清楚index_range, 在此项注明类型，会寻找匹配的index_range
        self.keep_level = True  # 是否保留level,该项在指定了location_type才有意义
        self.keep_patch_size = True  # 是否保留patch_size,该项在指定了location_type才有意义
        self.optional_operate = False  # 该项为False时，仅考虑将location，转为detect_location，不会有额外的separate操作
        self.separate_point_num = False  # 是否将注视点数量单独返回
        self.exist_point_num = True  # 该location是否包括注视点数量
        self.separate_mark = False  # 是否将mark数量单独返回
        self.exist_mark = True  # 该location是否包括mark
        # self.get_index_range()  # 计算匹配的index_range
        # self.check_keep_type()  # 检查是否满足keep_type=1的条件
        # self.check_optional_operate()  # 检查是否满足optional_operate=False的条件
        self.level_downsamples = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)

    def get_index_range(self):
        if self.location_type is not None:
            if self.location_type == "detect" or self.location_type == "detect_location":
                self.index_range[0] = 0
                self.get_index_range_1()
            elif self.location_type == "area" or self.location_type == "area_location":
                self.index_range[0] = 1
                self.get_index_range_1()
            elif self.location_type == "marked_area" or self.location_type == "marked_area_location":
                self.index_range[0] = 2
                self.get_index_range_1()
            elif self.location_type == "marked_detect" or self.location_type == "marked_detect":
                self.index_range[0] = 1
                self.get_index_range_1()
            else:
                raise TypeError(self.location_type + "in get_index_range")

    def get_index_range_1(self):
        if self.keep_level is True:
            if self.keep_patch_size is True:
                self.index_range[1] = None
            else:
                self.index_range[1] = -1
        else:
            self.index_range[1] = -2

    def check_keep_type(self):
        if self.index_range[0] == 0 or self.index_range[0] is None:
            if self.index_range[1] is None:
                if self.is_array is False or (self.is_array is True and self.is_level is False):
                    self.keep_type = True

    def check_optional_operate(self):
        if self.exist_mark is True and self.exist_point_num is True:
            if self.separate_point_num is True or self.separate_mark is True:
                pass
            else:
                self.optional_operate = False
        else:
            self.optional_operate = False

    def check_is_array(self, is_array=None):
        if is_array is not None:
            if is_array != self.is_array:
                self.is_array = is_array
        self.is_array_init = self.is_array
        self.get_index_range()  # 计算匹配的index_range
        self.check_keep_type()  # 检查是否满足keep_type=1的条件
        self.check_optional_operate()  # 检查是否满足optional_operate=False的条件

    def transform_detect_location(self, detect_location, is_array=None, grid_size=None, level_downsamples=None):
        self.check_is_array(is_array)
        separate_result = self.process_optional_operate(detect_location)
        detect_location = self.level2normal(detect_location)
        detect_location = self.grid_size_enlarge(detect_location, grid_size, level_downsamples)
        detect_location = self.other2detect(detect_location)
        self.is_array = self.is_array_init
        if separate_result is not None:
            return detect_location, separate_result
        else:
            return detect_location

    def grid_size_enlarge(self, detect_location, grid_size=None, level_downsamples=None):
        # 该函数仅适用于detect_location结构以level，patch_size结尾
        if grid_size is not None:
            if level_downsamples is not None:
                self.level_downsamples = level_downsamples
            detect_location = detect_location.copy()
            detect_location_new = []
            if self.is_array is True:
                for i in range(len(detect_location)):
                    for j in range(-grid_size[0], grid_size[0] + 1):
                        for k in range(-grid_size[1], grid_size[1] + 1):
                            location = detect_location[i].copy()
                            location[0] += int(
                                j * location[-1] * self.level_downsamples[location[-2]] /
                                self.level_downsamples[0])
                            location[1] += int(
                                k * location[-1] * self.level_downsamples[location[-2]] /
                                self.level_downsamples[0])
                            detect_location_new.append(location)
            else:
                for j in range(-grid_size[0], grid_size[0] + 1):
                    for k in range(-grid_size[1], grid_size[1] + 1):
                        detect_location_new_single = detect_location.copy()
                        detect_location_new_single[0] += int(
                            j * detect_location[-1] * self.level_downsamples[detect_location[-2]] /
                            self.level_downsamples[0])
                        detect_location_new_single[1] += int(
                            k * detect_location[-1] * self.level_downsamples[detect_location[-2]] /
                            self.level_downsamples[0])
                        detect_location_new.append(detect_location_new_single)
                self.is_array_init = self.is_array
                self.is_array = True
            detect_location = detect_location_new
        return detect_location

    def process_optional_operate(self, detect_location):
        if self.optional_operate is not False:
            mark_array = None
            point_num_array = None
            if self.exist_mark is True and self.separate_mark is True:
                mark_array = self.from_location_separate_mark(detect_location)
            if self.exist_point_num is True and self.separate_point_num is True:
                point_num_array = self.from_location_separate_point_num(detect_location)
            if mark_array is not None and point_num_array is not None:
                return mark_array, point_num_array
            elif mark_array is not None and point_num_array is None:
                return mark_array
            elif mark_array is None and point_num_array is not None:
                return point_num_array
            else:
                return None
        else:
            return None

    def from_location_separate_mark(self, detect_location):
        detect_location = detect_location.copy()
        if self.is_array is True:
            mark_array = [i[0] for i in detect_location]
        else:
            mark_array = detect_location[0]
        return mark_array

    def from_location_separate_point_num(self, detect_location):
        detect_location = detect_location.copy()
        if self.exist_point_num is True:
            if self.is_array is True:
                point_num_array = [i[1] for i in detect_location]
            else:
                point_num_array = detect_location[1]
        else:
            if self.is_array is True:
                point_num_array = [i[0] for i in detect_location]
            else:
                point_num_array = detect_location[1]
        return point_num_array

    def level2normal(self, detect_location):
        if self.is_level is True and self.is_array is True and self.keep_type is not True:
            detect_location_without_level = []
            for i in range(len(detect_location)):
                detect_location_without_level += detect_location[i]
            detect_location = detect_location_without_level
        return detect_location

    def other2detect(self, detect_location):
        detect_location = detect_location.copy()
        if self.keep_type is True:
            return detect_location
        else:
            if self.is_array is True:
                detect_location = [i[self.index_range[0]: self.index_range[1]] for i in detect_location]
            else:
                detect_location = detect_location[self.index_range[0]: self.index_range[1]]
            return detect_location
