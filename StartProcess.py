from produce_dataset import (GetInitDataset, DatasetRegularProcess)


class FullProcess(GetInitDataset, DatasetRegularProcess):
    def __init__(self, erc_path, record=True, func_mode=0, init_status=True, repeat_add_mode=True):
        super().__init__(erc_path, record, func_mode, init_status, repeat_add_mode)
        if self.config.is_save is False:
            self.information, self.result = None, None
            self.inner_process_flow()


    def inner_process_flow(self):
        self.path = self.name2path_in_list(self.path, self.config.image_path)
        self.information, self.result = self.process(self.name_array, self.path, self.point_array, self.level_array,
                                                     None, self.image_label, self.max_num, self.config)
        self.save(self.information, self.result, self.config.save_output_direc, self.config, self.config.save_mode)
        # self.config.is_save = True  # 该语句仅为测试save和read函数有效性

    def inner_produce_dataset_flow(self):
        # if self.config.is_save is not False:
        #     self.information, self.result = self.read(self.config.read_direc, self.config.read_mode)
        # read暂未实现
        dataset = self.produce_whole_dataset(self.information, self.result, self.config)
        train_dataset = dataset.produce_dataset(0, self.config.one_num, self.config.zero_num)
        val_dataset = dataset.produce_dataset(1, self.config.one_num, self.config.zero_num)
        return dataset, train_dataset, val_dataset




if __name__ == '__main__':
    process_func = FullProcess(r"D:\ajmq\point_cs", init_status=True)
