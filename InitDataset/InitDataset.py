from produce_dataset import DatasetRegularProcess


class InitDataset:
    def __init__(self, information, result, result_level, config):
        self.operate = DatasetRegularProcess()
        self.information = information
        self.all_patch = result
        self.result_level = result_level
        self.config = config

    def produce_dataset(self, index_use, one_num=None, zero_num=None):
        result = []
        total_one_num = 0
        total_zero_num = 0
        one_num_list = []
        zero_num_list = []
        for j in range(len(self.information['use_list'][index_use])):
            i = self.information['use_list'][index_use][j]
            slide = self.operate.read_slide(self.information['path'][j])
            name = self.information['name'][j]
            single_result = self.all_patch[index_use][i][0]
            marked_area_location, marked_zero_area_location = single_result[:2]
            zero_level_result, zero_result_reduce, zero_index_result, zero_num_result = self.operate.static_get_zero(
                marked_zero_area_location, self.result_level, zero_num, self.config)
            one_level_result, one_result_reduce, one_index_result, one_num_result = self.operate.static_get_one(
                marked_area_location, self.result_level, one_num, self.config)
            one_num_value = self.operate.static_sum_list_num(one_num_result)
            zero_num_value = self.operate.static_sum_list_num(zero_num_result)
            one_num_list.append(one_num_value)
            zero_num_list.append(zero_num_value)
            total_one_num += one_num_value
            total_zero_num += zero_num_value
            one_num = self.operate.static_calculate_num(one_result_reduce, self.config.calculate_one_num_mode)
            zero_num = self.operate.static_calculate_num(zero_result_reduce, self.config.calculate_zero_num_mode)
            result.append((slide, name, one_index_result, zero_index_result, one_level_result, zero_level_result))
        DatasetForLoader(index_use, result, total_one_num, total_zero_num, one_num_list, zero_num_list, self)

    def get_index


class DatasetForLoader:
    def __init__(self, index_use, result, one_num, zero_num, one_num_list, zero_num_list, init_dataset):
        self.class_num = index_use
        self.all_result = result
        self.one_num = one_num
        self.zero_num = zero_num
        self.one_num_list = one_num_list
        self.zero_num_list = zero_num_list
        self.one_num_add_list = self.calculate(one_num_list)
        self.zero_num_add_list = self.calculate(zero_num_list)
        self.length = one_num + zero_num
        self.base_dataset = init_dataset

    def calculate(self, num_list):
        num_list = num_list.copy()
        for i in range(len(num_list)):
            if i != 0:
                num_list[i] += num_list[i - 1]
        return num_list

    def __getitem__(self, idx):
        one_or_zero, position, index = self.get_position_index(idx)
        slide, name, one_index_result, zero_index_result, one_level_result, zero_level_result = self.all_result[position]





    def get_position_index(self, idx):
        position = -1
        index = -1
        if idx < self.one_num:
            one_or_zero = 1
            for i in range(len(self.one_num_add_list)):
                if i == 0:
                    if idx < self.one_num_add_list[i]:
                        position = i
                        index = idx
                else:
                    if self.one_num_add_list[i - 1] <= idx < self.one_num_add_list[i]:
                        position = i
                        index = idx - self.one_num_add_list[i - 1]
        else:
            one_or_zero = 0
            for i in range(len(self.zero_num_add_list)):
                if i == 0:
                    if idx < self.one_num_add_list[i]:
                        position = i
                        index = idx - self.one_num
                else:
                    if self.one_num_add_list[i - 1] <= idx < self.one_num_add_list[i]:
                        position = i
                        index = idx - self.one_num_add_list[i - 1] - self.one_num
        return one_or_zero, position, index

    def get_index_in_index_result(self, index, one_or_zero, one_index_result, zero_index_result):
        if one_or_zero == 0:
            position, index_result = self.get_index_in_index_result_process(index, zero_index_result)
        else:
            position, index_result = self.get_index_in_index_result_process(index, one_index_result)

    def get_index_in_index_result_process(self, index, class_index_result):
        position = -1
        index_result = -1
        add_list = self.produce_add_list(class_index_result)
        for i in range(len(add_list)):
            if i == 0:
                if index < add_list[i]:
                    position = i
                    index_result = index
            else:
                if add_list[i - 1] <= index < add_list[i]:
                    position = i
                    index_result = index - add_list[i - 1]
        return position, index_result

    def produce_add_list(self, list_a):
        add_list = []
        for i in range(len(list_a)):
            if i == 0:
                add_list.append(len(list_a))
            else:
                add_list.append(len(list_a) + add_list[i - 1])
        return add_list

    def __len__(self):
        return self.length
