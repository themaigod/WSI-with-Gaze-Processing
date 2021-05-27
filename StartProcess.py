from produce_dataset import (GetInitDataset, DatasetRegularProcess)
from torch.utils.data import DataLoader
import time
import json
import os


# 目前处理方法存在一点点问题，即没有考虑块旁边的块有可能在背景上，对模型带来脏数据

class FullProcess(GetInitDataset, DatasetRegularProcess):
    def __init__(self, erc_path, record=True, func_mode=0, init_status=True, repeat_add_mode=True):
        super().__init__(erc_path, record, func_mode, init_status, repeat_add_mode)
        if self.config.is_save is False:
            time_start = time.time()
            self.information, self.result = None, None
            self.inner_process_flow()
            print("process time:" + str(time.time() - time_start))
        self.dataset = None
        print("start build base dataset")
        time_start = time.time()
        self.inner_produce_dataset_flow()
        print("build base dataset time:" + str(time.time() - time_start))
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        # self.inner_get_output_dataset()

    def inner_process_flow(self):
        self.path = self.name2path_in_list(self.path, self.config.image_path)
        self.information, self.result = self.process(self.name_array, self.path, self.point_array, self.level_array,
                                                     None, self.image_label, self.max_num, self.config)
        self.save(self.information, self.result, self.config.save_output_direc, self.config, self.config.save_mode)
        self.config.is_save = True  # 该语句仅为测试save和read函数有效性

    def inner_produce_dataset_flow(self):
        # if self.config.is_save is not False:
        #     self.information, self.result = self.read(self.config.read_direc, self.config.read_mode)
        # read暂未实现
        if self.config.is_save is not False:
            self.information, self.result = self.read(self.config.read_direc)
        # 使用暂时的read方法，主要因为前面步骤有点花时间
        self.dataset = self.produce_whole_dataset(self.information, self.result, self.config)
        return self.dataset

    def read(self, direc, mode=0):
        # 使用暂时的read方法，主要因为前面步骤有点花时间
        with open(os.path.join(direc, "patch.json")) as f:
            patches = json.load(f)
        with open(os.path.join(direc, "information.json")) as g:
            information = json.load(g)
        return information, patches

    def inner_get_output_dataset(self, num=None):
        if num is None:
            time_start = time.time()
            print("start build train dataset")
            self.train_dataset = self.dataset.produce_dataset(0, self.config.one_num, self.config.zero_num)
            print("build train dataset time:" + str(time.time() - time_start))
            print("start build val dataset")
            time_start = time.time()
            self.val_dataset = self.dataset.produce_dataset(1, self.config.one_num, self.config.zero_num)
            print("build val dataset time:" + str(time.time() - time_start))
            return self.train_dataset, self.val_dataset
        elif num == 0:
            self.train_dataset = self.dataset.produce_dataset(0, self.config.one_num, self.config.zero_num)
            return self.train_dataset
        elif num == 1:
            self.val_dataset = self.dataset.produce_dataset(1, self.config.one_num, self.config.zero_num)
            return self.val_dataset

    def inner_get_output_mil_dataset(self, num=None):
        # 暂行方案
        if num is None:
            time_start = time.time()
            print("start build train total dataset")
            self.train_dataset = self.dataset.produce_dataset_mil_total(0)
            print("build train dataset time:" + str(time.time() - time_start))
            print("start build val dataset")
            time_start = time.time()
            self.val_dataset = self.dataset.produce_dataset_mil_total(1)
            print("build val dataset time:" + str(time.time() - time_start))
            return self.train_dataset, self.val_dataset
        elif num == 2:
            time_start = time.time()
            print("start build train total dataset")
            self.test_dataset = self.dataset.produce_dataset_mil_total(2)
            print("build train dataset time:" + str(time.time() - time_start))
            return self.test_dataset


if __name__ == '__main__':
    process_func = FullProcess(r"/home/omnisky/ajmq/process_operate_relate/point", init_status=True)
    process_func.inner_get_output_dataset()
    # train_loader = DataLoader(process_func.train_dataset, batch_size=10, num_workers=12, shuffle=True, drop_last=True)
    now_time = time.time()
    # for step, (img, label, patch, position) in enumerate(train_loader):
    #     # print(img)
    #     print(time.time() - now_time)
    #     now_time = time.time()
    #     # print(label)
