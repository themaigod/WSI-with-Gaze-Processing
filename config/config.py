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
        self.group_mark_mode = 1
        self.zero_ratio = None
        self.zero_num_mode = 4
        self.get_zero_index_mode = 0
