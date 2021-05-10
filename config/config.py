class Config:  # 初始化参数
    def __init__(self):
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
        self.calculate_one_num_mode = 0
        self.calculate_zero_num_mode = 0
        self.class_ratio = (1, 1, 1)
        self.image_path = r"D:\ajmq\data_no_process"
        self.information_save = "json"
        # 还支持"numpy" "mat" "pickle"(官方标准包）  "key" 代表按key分别按numpy储存 #如果存在不匹配的，按key储存
        self.result_save = "json"
        # 还支持"numpy" "mat" "pickle"(官方标准包， 该方法暂定储存类型为pkl，其实支持write接口的对象均可使用）
        # 如果如果存在不匹配的，按numpy储存
