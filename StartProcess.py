from produce_dataset import (GetInitDataset, DatasetRegularProcess)


class FullProcess(GetInitDataset, DatasetRegularProcess):
    def __init__(self, erc_path, record=True, func_mode=0, init_status=True, repeat_add_mode=True):
        super().__init__(erc_path, record, func_mode, init_status, repeat_add_mode)
        self.inner_process_flow()
        self.information, self.result, self.class_one_num, self.class_zero_num = None, None, None, None

    def inner_process_flow(self):
        self.path = self.name2path_in_list(self.path, self.config.image_path)
        self.information, self.result, self.class_one_num, self.class_zero_num = self.process(self.name_array,
                                                                                              self.path,
                                                                                              self.point_array,
                                                                                              self.level_array,
                                                                                              None, self.image_label,
                                                                                              self.max_num,
                                                                                              self.config)


if __name__ == '__main__':
    process_func = FullProcess(r"D:\ajmq\point_cs", init_status=True)
