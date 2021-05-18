from produce_dataset import (GetInitDataset, DatasetRegularProcess)
from torch.utils.data import DataLoader


class FullProcess(GetInitDataset, DatasetRegularProcess):
    def __init__(self, erc_path, record=True, func_mode=0, init_status=True, repeat_add_mode=True):
        super().__init__(erc_path, record, func_mode, init_status, repeat_add_mode)
        if self.config.is_save is False:
            self.information, self.result = None, None
            self.inner_process_flow()
        self.dataset = None
        self.inner_produce_dataset_flow()
        self.train_dataset = None
        self.val_dataset = None
        # self.inner_get_output_dataset()

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
        self.dataset = self.produce_whole_dataset(self.information, self.result, self.config)
        return self.dataset

    def inner_get_output_dataset(self, num=None):
        if num is None:
            self.train_dataset = self.dataset.produce_dataset(0, self.config.one_num, self.config.zero_num)
            self.val_dataset = self.dataset.produce_dataset(1, self.config.one_num, self.config.zero_num)
            return self.train_dataset, self.val_dataset
        elif num == 0:
            self.train_dataset = self.dataset.produce_dataset(0, self.config.one_num, self.config.zero_num)
            return self.train_dataset
        elif num == 1:
            self.val_dataset = self.dataset.produce_dataset(1, self.config.one_num, self.config.zero_num)
            return self.val_dataset


if __name__ == '__main__':
    process_func = FullProcess(r"/home/omnisky/ajmq/process_operate_relate/point_test", init_status=True)
    process_func.inner_get_output_dataset()
    train_loader = DataLoader(process_func.train_dataset, batch_size=10, num_workers=2, shuffle=True, drop_last=True)
    for step, (img, label, patch, position) in enumerate(train_loader):
        # print(img)
        print(label)
