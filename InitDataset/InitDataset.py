import numpy as np
import cv2
import copy


class InitDataset:
    def __init__(self, information, result, config, instance):
        self.operate = instance
        self.information = information
        self.all_patch = result
        self.config = config
        self.result_record = []
        self.record_result_patch()

    def record_result_patch(self):
        for i in range(len(self.all_patch)):
            class_slide_record = []
            for j in range(len(self.all_patch[i])):
                one_record = [[] for _ in range(len(self.all_patch[i][j][0][0]))]
                zero_record = [[] for _ in range(len(self.all_patch[i][j][0][1]))]
                class_slide_record.append([one_record, zero_record])
            self.result_record.append(class_slide_record)

    def produce_dataset(self, index_use, one_num=None, zero_num=None):
        result = []
        total_one_num = 0
        total_zero_num = 0
        one_num_list = []
        zero_num_list = []
        for i in range(len(self.information['use_list'][index_use])):
            j = self.information['use_list'][index_use][i]
            slide = self.operate.read_slide(self.information['path'][j])
            name = self.information['name'][j]
            single_result = self.all_patch[index_use][i][0]
            marked_area_location, marked_zero_area_location, result_level = single_result

            if index_use == 0:
                config = copy.deepcopy(self.config)
                config.get_zero_index_mode = 1
            else:
                config = self.config
            # 临时处理，暂时采用的方法，训练时依照mark选patch

            zero_level_result, zero_result_reduce, zero_index_result, zero_num_result = self.operate.static_get_zero(
                marked_zero_area_location, result_level, zero_num, config)
            one_level_result, one_result_reduce, one_index_result, one_num_result = self.operate.static_get_one(
                marked_area_location, result_level, one_num, config)
            one_num_value = self.operate.static_sum_list_num(one_num_result)
            zero_num_value = self.operate.static_sum_list_num(zero_num_result)
            one_num_list.append(one_num_value)
            zero_num_list.append(zero_num_value)
            total_one_num += one_num_value
            total_zero_num += zero_num_value
            one_num = self.operate.static_calculate_num(one_result_reduce, self.config.calculate_one_num_mode)
            zero_num = self.operate.static_calculate_num(zero_result_reduce, self.config.calculate_zero_num_mode)
            result.append((self.information['path'][j], name, one_index_result, zero_index_result, one_level_result, zero_level_result))
        return DatasetForLoader(index_use, result, total_one_num, total_zero_num, one_num_list, zero_num_list, self)

    def get_index(self, result, patch, position, index_use):
        # 该方案有点效率低下，主要因为level_area_location不能转化为area_location, 需要area_location转level_area_location时返回坐标关系才能直接映射
        # 所以这里直接输入patch，运用list的index的方法寻找坐标
        for i in range(len(result)):
            [slide_index, one_or_zero] = position[i]
            index = self.all_patch[index_use][slide_index][0][int(1 - one_or_zero)].index(patch[i])
            self.result_record[index_use][slide_index][int(1 - one_or_zero)][index].append(result[i])


class DatasetForLoader:
    def __init__(self, index_use, result, one_num, zero_num, one_num_list, zero_num_list, init_dataset: InitDataset):
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
        self.use_random = False
        self._patch_per_side = int(self.base_dataset.config.grid_size[0] * 2 + 1)

    def is_train(self, is_train=True):
        self.use_random = is_train

    @staticmethod
    def calculate(num_list):
        num_list = num_list.copy()
        for i in range(len(num_list)):
            if i != 0:
                num_list[i] += num_list[i - 1]
        return num_list

    def __getitem__(self, idx):
        one_or_zero, position, index = self.get_position_index(idx)
        slide_name, name, one_index_result, zero_index_result, one_level_result, zero_level_result = self.all_result[
            position]
        slide = self.base_dataset.operate.read_slide(slide_name)
        index_position, index_index = self.get_index_in_index_result(index, one_or_zero, one_index_result,
                                                                     zero_index_result)
        if one_or_zero == 1:
            true_index = one_index_result[index_position][index_index]
            patch = one_level_result[index_position][true_index]
        else:
            true_index = zero_index_result[index_position][index_index]
            patch = zero_level_result[index_position][true_index]
        [x, y, level, patch_size] = patch[2:]
        idx = x - self.base_dataset.operate.static_double_mul_div_int_mul_level(
            self.base_dataset.config.grid_size[0] * patch_size,
            slide.level_downsamples, 0, level)
        idy = y - self.base_dataset.operate.static_double_mul_div_int_mul_level(
            self.base_dataset.config.grid_size[1] * patch_size,
            slide.level_downsamples, 0, level)
        x_len = patch_size * (1 + 2 * self.base_dataset.config.grid_size[0])
        y_len = patch_size * (1 + 2 * self.base_dataset.config.grid_size[1])
        img: np.ndarray
        img = self.base_dataset.operate.read_image_area(slide, (int(idx), int(idy)), level, (int(x_len), int(y_len)),
                                                        True)[:, :, :3]

        label = self.base_dataset.operate.set_label([x, y, level, patch_size], one_level_result,
                                                    self.base_dataset.config,
                                                    self.base_dataset.config.set_label_in_dataset_for_loader_control,
                                                    self.base_dataset.config.grid_size, slide.level_downsamples)

        label = np.array(label, np.float32)

        label_grid = np.zeros((self._patch_per_side, self._patch_per_side),
                              dtype=np.float32)

        length = 0
        for x_idx in range(self._patch_per_side):
            for y_idx in range(self._patch_per_side):
                label_grid[x_idx, y_idx] = label[length]
                length = length + 1

        if self.use_random is True:
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
                label_grid = np.fliplr(label_grid)
            if np.random.rand() > 0.5:
                num_rotate = np.random.randint(0, 4)
                matRotate = cv2.getRotationMatrix2D((img.shape[1] * 0.5, img.shape[0] * 0.5), 90 * num_rotate, 1.0)
                img = cv2.warpAffine(img, matRotate, (img.shape[1], img.shape[1]))
                label_grid = np.rot90(label_grid, num_rotate)
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))
        img = (img - 128.0) / 128.0

        grid_size = (2 * self.base_dataset.config.grid_size[0] + 1) * (2 * self.base_dataset.config.grid_size[1] + 1)
        img_flat = np.zeros(
            (grid_size, 3, self.base_dataset.config.crop_size, self.base_dataset.config.crop_size),
            dtype=np.float32)
        label_flat = np.zeros(grid_size, dtype=np.float32)

        idn = 0
        for x_idx in range(self.base_dataset.config.grid_size[0] * 2 + 1):
            for y_idx in range(self.base_dataset.config.grid_size[1] * 2 + 1):
                # center crop each patch
                x_start = int(
                    (x_idx + 0.5) * self.base_dataset.config.patch_size - self.base_dataset.config.crop_size / 2)
                x_end = x_start + self.base_dataset.config.crop_size
                y_start = int(
                    (y_idx + 0.5) * self.base_dataset.config.patch_size - self.base_dataset.config.crop_size / 2)
                y_end = y_start + self.base_dataset.config.crop_size
                img_flat[idn] = img[:, x_start:x_end, y_start:y_end]
                label_flat[idn] = label_grid[x_idx, y_idx]
                idn += 1
        return img_flat, label_flat, np.array(patch), np.array([position, one_or_zero])

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
                    if idx - self.one_num < self.zero_num_add_list[i]:
                        position = i
                        index = idx - self.one_num
                else:
                    if self.zero_num_add_list[i - 1] <= idx - self.one_num < self.zero_num_add_list[i]:
                        position = i
                        index = idx - self.zero_num_add_list[i - 1] - self.one_num
        return one_or_zero, position, index

    def get_index_in_index_result(self, index, one_or_zero, one_index_result, zero_index_result):
        if one_or_zero == 0:
            position, index_result = self.get_index_in_index_result_process(index, zero_index_result)
        else:
            position, index_result = self.get_index_in_index_result_process(index, one_index_result)
        return position, index_result

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

    @staticmethod
    def produce_add_list(list_a):
        add_list = []
        for i in range(len(list_a)):
            if i == 0:
                add_list.append(len(list_a))
            else:
                add_list.append(len(list_a) + add_list[i - 1])
        return add_list

    def __len__(self):
        return self.length
