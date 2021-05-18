from .set_label import (Control, DetectLocation, Point)


class Config:  # 初始化参数
    def __init__(self):
        # grid设置
        self.grid_size = (1, 1)  # 上下及左右延长距离 例如（2，5）代表的大小为（1 + 2 * 2， 1 + 5 * 2）

        # process设置
        self.use_level_array = True
        self.use_start_point = False
        self.use_area_size = False
        self.start_point = (0, 0)
        self.area_size = None  # example: (1000, 1000)
        self.threshold = 0
        self.distance_mode = 0
        self.point_mark_mode = 0
        self.detect_edge = True
        self.group_mark_mode = 0
        self.zero_ratio = 0.9
        self.one_ratio = 0.9
        self.zero_num_mode = 4
        self.one_num_mode = 2
        self.get_zero_index_mode = 1
        self.level = 3
        self.level_img = 5
        self.patch_size = 256
        self.class_ratio = (7, 1, 2)

        # 是否使用多进程
        self.multi_process = True
        self.multi_process_num = 6

        #
        self.calculate_one_num_mode = 0
        self.calculate_zero_num_mode = 0

        # 获取数据设置
        self.image_path = r"/home/omnisky/ajmq/slideclassify/data_no_process"

        # 保存设置
        self.information_save = "json"
        # 还支持"numpy" "mat" "pickle"(官方标准包）  "key" 代表按key分别按numpy储存 #如果存在不匹配的，按key储存
        self.result_save = "json"
        # 还支持"numpy" "mat" "pickle"(官方标准包， 该方法暂定储存类型为pkl，其实支持write接口的对象均可使用）
        # 如果如果存在不匹配的，按numpy储存
        self.save_output_direc = r"/home/omnisky/ajmq/process_operate_relate/result"
        self.save_mode = 0

        # static_read_save_patch相关设置
        self.read_point = Point(True)
        self.read_point.keep_patch_size = True

        self.set_label_control = Control()

        # 是否保存设置
        self.is_save = False  # 是否有保存

        # 读取设置，注：read函数暂未实现
        self.read_direc = ""
        self.read_mode = 0

        # set_label使用设置
        self.set_label_in_dataset_for_loader_control = Control(False)
        self.set_label_in_dataset_for_loader_control.detect_location.location_type = "marked_area_location"
        self.set_label_in_dataset_for_loader_control.point.exist_mark = False
        self.set_label_in_dataset_for_loader_control.point.exist_point_num = False
        self.set_label_in_dataset_for_loader_control.point.location_type = "detect_location"
        self.middle_value = 0.5
        self.detect_surround = False

        # inner_get_output_dataset
        self.zero_num = None
        self.one_num = None
