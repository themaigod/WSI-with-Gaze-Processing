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
        self.keep_patch_size = False  # 是否保留patch_size,该项在指定了location_type才有意义
        self.optional_operate = False  # 该项为False时，仅考虑将location，转为detect_location，不会有额外的separate操作
        self.separate_point_num = False  # 是否将注视点数量单独返回
        self.exist_point_num = True  # 该location是否包括注视点数量
        self.separate_mark = False  # 是否将mark数量单独返回
        self.exist_mark = True  # 该location是否包括mark
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
        separate_result = self.process_optional_operate(detect_location)
        detect_location = self.other2detect(detect_location)
        if separate_result is not None:
            return detect_location, separate_result
        else:
            return detect_location


class Point:
    def __init__(self, is_array=True):
        self.keep_type = False  # 不改变detect_location，直接返回
        self.index_range = [0, None]  # 取的location的index范围, 如果想依靠location_type获得，建议填写[0, None]
        self.is_array = is_array
        self.is_level = True  # 是否是按level储存, 该项仅在is_array=True有效
        self.location_type = None  # 如果不清楚index_range, 在此项注明类型，会寻找匹配的index_range
        self.keep_level = True  # 是否保留level,该项在指定了location_type才有意义
        self.keep_patch_size = False  # 是否保留patch_size,该项在指定了location_type才有意义
        self.optional_operate = False  # 该项为False时，仅考虑将location，转为detect_location，不会有额外的separate操作
        self.separate_point_num = False  # 是否将注视点数量单独返回
        self.exist_point_num = True  # 该location是否包括注视点数量
        self.separate_mark = False  # 是否将mark数量单独返回
        self.exist_mark = True  # 该location是否包括mark
        self.get_index_range()  # 计算匹配的index_range
        self.check_keep_type()  # 检查是否满足keep_type=1的条件
        self.check_optional_operate()  # 检查是否满足optional_operate=False的条件

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

    def transform_detect_location(self, detect_location):
        separate_result = self.process_optional_operate(detect_location)
        detect_location = self.other2detect(detect_location)
        if separate_result is not None:
            return detect_location, separate_result
        else:
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

    def other2detect(self, detect_location):
        detect_location = detect_location.copy()
        if self.keep_type is True:
            return detect_location
        else:
            if self.is_level is True and self.is_array is True:
                detect_location_without_level = []
                for i in range(len(detect_location)):
                    detect_location_without_level += detect_location[i]
                detect_location = detect_location_without_level
            if self.is_array is True:
                detect_location = [i[self.index_range[0]: self.index_range[1]] for i in detect_location]
            else:
                detect_location = detect_location[self.index_range[0]: self.index_range[1]]
            return detect_location
