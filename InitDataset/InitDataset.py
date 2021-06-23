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
        self.slide_record = []
        self.result_max = [[], [], []]
        self.record_result_slide()

    def record_result_patch(self):
        for i in range(len(self.all_patch)):
            class_slide_record = []
            for j in range(len(self.all_patch[i])):
                one_record = [[] for _ in range(len(self.all_patch[i][j][0][0]))]
                zero_record = [[] for _ in range(len(self.all_patch[i][j][0][1]))]
                class_slide_record.append([one_record, zero_record])
            self.result_record.append(class_slide_record)

    def record_result_slide(self):
        for i in range(len(self.all_patch)):
            class_slide_record = []
            for j in range(len(self.all_patch[i])):
                one_record = [[] for _ in range(len(self.all_patch[i][j][0][0]))]
                zero_record = [[] for _ in range(len(self.all_patch[i][j][0][1]))]
                total_record = [one_record, zero_record]
                class_slide_record.append(total_record)
            self.slide_record.append(class_slide_record)

    def produce_dataset(self, index_use, one_num=None, zero_num=None):
        result = []
        total_one_num = 0
        total_zero_num = 0
        one_num_list = []
        zero_num_list = []
        for i in range(len(self.information['use_list'][index_use])):
            j = self.information['use_list'][index_use][i]
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
            result.append((self.information['path'][j], name, one_index_result, zero_index_result, one_level_result,
                           zero_level_result))
        return DatasetForLoader(index_use, result, total_one_num, total_zero_num, one_num_list, zero_num_list, self)

    def slide_max(self, index_use):
        class_result = self.slide_record[index_use]
        result_max = []
        for i in range(len(class_result)):
            [one_record, zero_record] = class_result[i]
            len_one = len(one_record)
            len_zero = len(zero_record)
            result_slide = []
            for j in range(len_one):
                try:
                    mark = one_record[j][0]
                except:
                    mark = -100
                result_slide.append(mark)
            for k in range(len_zero):
                try:
                    mark = zero_record[k][0]
                except:
                    mark = -100
                result_slide.append(mark)
            index_data = [[index, value] for index, value in
                          sorted(enumerate(result_slide), key=lambda x: x[1], reverse=True)]
            index_data_one_zero = [[1, m[0] - len_one, m[1]] if m[0] >= len_one else [0, m[0], m[1]] for m in
                                   index_data]
            result_max.append(index_data_one_zero)
        self.result_max[index_use] = result_max

    def produce_dataset_mil_total(self, index_use, one_num=None, zero_num=None):
        result = []
        total_one_num = 0
        total_zero_num = 0
        one_num_list = []
        zero_num_list = []
        for i in range(len(self.information['use_list'][index_use])):
            j = self.information['use_list'][index_use][i]
            name = self.information['name'][j]
            single_result = self.all_patch[index_use][i][0]
            marked_area_location, marked_zero_area_location, result_level = single_result
            if index_use == 0:
                config = copy.deepcopy(self.config)
                config.get_zero_index_mode = 1
            else:
                config = self.config
            # 临时处理，暂时采用的方法，训练时依照mark选patch

            # zero_level_result, zero_result_reduce, zero_index_result, zero_num_result = self.operate.static_get_zero(
            #     marked_zero_area_location, result_level, zero_num, config)
            # one_level_result, one_result_reduce, one_index_result, one_num_result = self.operate.static_get_one(
            #     marked_area_location, result_level, one_num, config)

            zero_level_result, zero_num_level, zero_level = self.operate.static_zero_list2zero_list_level(
                marked_zero_area_location)
            one_level_result, one_num_level, one_level = self.operate.static_zero_list2zero_list_level(
                marked_area_location)

            # for p in range(len(marked_area_location)):
            #     if len(marked_area_location[p]) == 5:
            #         lo = marked_area_location[p]
            #         print(marked_area_location[p])
            # for p in range(len(marked_zero_area_location)):
            #     if len(marked_zero_area_location[p]) == 5:
            #         lo = marked_zero_area_location[p]
            #         print(marked_zero_area_location[p])

            one_index_result = [list(range(len(one_level_result[k]))) for k in range(len(one_level_result))]
            zero_index_result = [list(range(len(zero_level_result[m]))) for m in range(len(zero_level_result))]
            one_num_result = [len(one_index_result[n]) for n in range(len(one_index_result))]
            zero_num_result = [len(zero_index_result[n]) for n in range(len(zero_index_result))]

            one_num_value = self.operate.static_sum_list_num(one_num_result)
            zero_num_value = self.operate.static_sum_list_num(zero_num_result)
            one_num_list.append(one_num_value)
            zero_num_list.append(zero_num_value)
            total_one_num += one_num_value
            total_zero_num += zero_num_value
            result.append((self.information['path'][j], name, one_index_result, zero_index_result, one_level_result,
                           zero_level_result, i, j))
        return DatasetForLoaderMIL(index_use, result, total_one_num, total_zero_num, one_num_list, zero_num_list, self)

    def produce_dataset_mil(self, index_use, top_k=5):
        result = []
        total_one_num = 0
        total_zero_num = 0
        one_num_list = []
        zero_num_list = []
        for i in range(len(self.information['use_list'][index_use])):
            j = self.information['use_list'][index_use][i]
            name = self.information['name'][j]
            single_result = self.all_patch[index_use][i][0]
            marked_area_location, marked_zero_area_location, result_level = single_result

            if index_use == 0:
                config = copy.deepcopy(self.config)
                config.get_zero_index_mode = 1
            else:
                config = self.config
            # 临时处理，暂时采用的方法，训练时依照mark选patch

            # zero_level_result, zero_result_reduce, zero_index_result, zero_num_result = self.operate.static_get_zero(
            #     marked_zero_area_location, result_level, zero_num, config)
            # one_level_result, one_result_reduce, one_index_result, one_num_result = self.operate.static_get_one(
            #     marked_area_location, result_level, one_num, config)

            zero_level_result, zero_num_level, zero_level = self.operate.static_zero_list2zero_list_level(
                marked_zero_area_location)
            one_level_result, one_num_level, one_level = self.operate.static_zero_list2zero_list_level(
                marked_area_location)
            result_max_slide = self.result_max[index_use][i]
            label_mil = np.array([np.float32(self.information['label'][j])])
            label_non = np.array([np.float32(0)])
            one_index_result_exist = [[] for _ in range(len(one_level_result))]
            zero_index_result_exist = [[] for _ in range(len(zero_level_result))]
            one_index_result_non_exist = [[] for _ in range(len(one_level_result))]
            zero_index_result_non_exist = [[] for _ in range(len(zero_level_result))]
            one_index_result = [[] for _ in range(len(one_level_result))]
            zero_index_result = [[] for _ in range(len(zero_level_result))]
            one_exist_label_mil = [[] for _ in range(len(one_level_result))]
            zero_exist_label_mil = [[] for _ in range(len(zero_level_result))]
            one_non_exist_label_mil = [[] for _ in range(len(one_level_result))]
            zero_non_exist_label_mil = [[] for _ in range(len(zero_level_result))]
            one_label_mil = [[] for _ in range(len(one_level_result))]
            zero_label_mil = [[] for _ in range(len(zero_level_result))]
            for k in range(top_k):
                try:
                    result_use = result_max_slide[k]
                    if result_use[0] == 0:
                        patch = marked_area_location[result_use[1]]
                        for m in range(len(one_level_result)):
                            if patch in one_level_result[m]:
                                one_index_result_exist[m].append(one_level_result[m].index(patch))
                                one_exist_label_mil[m].append(label_mil)
                    else:
                        patch = marked_zero_area_location[result_use[1]]
                        for m in range(len(zero_level_result)):
                            if patch in zero_level_result[m]:
                                zero_index_result_exist[m].append(zero_level_result[m].index(patch))
                                zero_exist_label_mil[m].append(label_mil)
                finally:
                    pass

                try:
                    for shift in range(len(result_max_slide)):
                        if result_max_slide[-k - shift][2] != -100:
                            result_use = result_max_slide[-k - shift]
                            if result_use[0] == 0:
                                patch = marked_area_location[result_use[1]]
                                for m in range(len(one_level_result)):
                                    if patch in one_level_result[m]:
                                        one_index_result_non_exist[m].append(one_level_result[m].index(patch))
                                        one_non_exist_label_mil[m].append(label_non)
                            else:
                                patch = marked_zero_area_location[result_use[1]]
                                for m in range(len(zero_level_result)):
                                    if patch in zero_level_result[m]:
                                        zero_index_result_non_exist[m].append(zero_level_result[m].index(patch))
                                        zero_non_exist_label_mil[m].append(label_non)
                            break
                finally:
                    pass
            for n in range(len(one_level_result)):
                one_index_result[n] = one_index_result_exist[n] + one_index_result_non_exist[n]
                one_label_mil[n] = one_exist_label_mil[n] + one_non_exist_label_mil[n]

            for n in range(len(zero_level_result)):
                zero_index_result[n] = zero_index_result_exist[n] + zero_index_result_non_exist[n]
                zero_label_mil[n] = zero_exist_label_mil[n] + zero_non_exist_label_mil[n]

            one_num_result = [len(one_index_result[n]) for n in range(len(one_index_result))]
            zero_num_result = [len(zero_index_result[n]) for n in range(len(zero_index_result))]

            one_num_value = self.operate.static_sum_list_num(one_num_result)
            zero_num_value = self.operate.static_sum_list_num(zero_num_result)
            one_num_list.append(one_num_value)
            zero_num_list.append(zero_num_value)
            total_one_num += one_num_value
            total_zero_num += zero_num_value
            result.append((self.information['path'][j], name, one_index_result, zero_index_result, one_level_result,
                           zero_level_result, i, j, one_label_mil, zero_label_mil))
        return DatasetForLoaderMIL(index_use, result, total_one_num, total_zero_num, one_num_list, zero_num_list, self,
                                   for_train=True)

    def produce_dataset_test_mil(self, index_use, top_k=5):
        result = []
        total_one_num = 0
        total_zero_num = 0
        one_num_list = []
        zero_num_list = []
        for i in range(len(self.information['use_list'][index_use])):
            j = self.information['use_list'][index_use][i]
            name = self.information['name'][j]
            single_result = self.all_patch[index_use][i][0]
            marked_area_location, marked_zero_area_location, result_level = single_result

            if index_use == 0:
                config = copy.deepcopy(self.config)
                config.get_zero_index_mode = 1
            else:
                config = self.config
            # 临时处理，暂时采用的方法，训练时依照mark选patch

            # zero_level_result, zero_result_reduce, zero_index_result, zero_num_result = self.operate.static_get_zero(
            #     marked_zero_area_location, result_level, zero_num, config)
            # one_level_result, one_result_reduce, one_index_result, one_num_result = self.operate.static_get_one(
            #     marked_area_location, result_level, one_num, config)

            zero_level_result, zero_num_level, zero_level = self.operate.static_zero_list2zero_list_level(
                marked_zero_area_location)
            one_level_result, one_num_level, one_level = self.operate.static_zero_list2zero_list_level(
                marked_area_location)
            result_max_slide = self.result_max[index_use][i]
            label_mil = np.array([np.float32(self.information['label'][j])])
            label_non = np.array([np.float32(0)])
            one_index_result_exist = [[] for _ in range(len(one_level_result))]
            zero_index_result_exist = [[] for _ in range(len(zero_level_result))]
            one_index_result_non_exist = [[] for _ in range(len(one_level_result))]
            zero_index_result_non_exist = [[] for _ in range(len(zero_level_result))]
            one_index_result = [[] for _ in range(len(one_level_result))]
            zero_index_result = [[] for _ in range(len(zero_level_result))]
            one_exist_label_mil = [[] for _ in range(len(one_level_result))]
            zero_exist_label_mil = [[] for _ in range(len(zero_level_result))]
            one_non_exist_label_mil = [[] for _ in range(len(one_level_result))]
            zero_non_exist_label_mil = [[] for _ in range(len(zero_level_result))]
            one_label_mil = [[] for _ in range(len(one_level_result))]
            zero_label_mil = [[] for _ in range(len(zero_level_result))]
            for k in range(top_k):
                try:
                    result_use = result_max_slide[k]
                    if result_use[0] == 0:
                        patch = marked_area_location[result_use[1]]
                        for m in range(len(one_level_result)):
                            if patch in one_level_result[m]:
                                one_index_result_exist[m].append(one_level_result[m].index(patch))
                                one_exist_label_mil[m].append(label_mil)
                    else:
                        patch = marked_zero_area_location[result_use[1]]
                        for m in range(len(zero_level_result)):
                            if patch in zero_level_result[m]:
                                zero_index_result_exist[m].append(zero_level_result[m].index(patch))
                                zero_exist_label_mil[m].append(label_mil)
                finally:
                    pass

                # try:
                #     for shift in range(len(result_max_slide)):
                #         if result_max_slide[-k - shift][2] != -100:
                #             result_use = result_max_slide[-k - shift]
                #             if result_use[0] == 0:
                #                 patch = marked_area_location[result_use[1]]
                #                 for m in range(len(one_level_result)):
                #                     if patch in one_level_result[m]:
                #                         one_index_result_non_exist[m].append(one_level_result[m].index(patch))
                #                         one_non_exist_label_mil[m].append(label_non)
                #             else:
                #                 patch = marked_zero_area_location[result_use[1]]
                #                 for m in range(len(zero_level_result)):
                #                     if patch in zero_level_result[m]:
                #                         zero_index_result_non_exist[m].append(zero_level_result[m].index(patch))
                #                         zero_non_exist_label_mil[m].append(label_non)
                #             break
                # finally:
                #     pass
            for n in range(len(one_level_result)):
                one_index_result[n] = one_index_result_exist[n] + one_index_result_non_exist[n]
                one_label_mil[n] = one_exist_label_mil[n] + one_non_exist_label_mil[n]

            for n in range(len(zero_level_result)):
                zero_index_result[n] = zero_index_result_exist[n] + zero_index_result_non_exist[n]
                zero_label_mil[n] = zero_exist_label_mil[n] + zero_non_exist_label_mil[n]

            one_num_result = [len(one_index_result[n]) for n in range(len(one_index_result))]
            zero_num_result = [len(zero_index_result[n]) for n in range(len(zero_index_result))]

            one_num_value = self.operate.static_sum_list_num(one_num_result)
            zero_num_value = self.operate.static_sum_list_num(zero_num_result)
            one_num_list.append(one_num_value)
            zero_num_list.append(zero_num_value)
            total_one_num += one_num_value
            total_zero_num += zero_num_value
            result.append((self.information['path'][j], name, one_index_result, zero_index_result, one_level_result,
                           zero_level_result, i, j, one_label_mil, zero_label_mil))
        return DatasetForLoaderMIL(index_use, result, total_one_num, total_zero_num, one_num_list, zero_num_list, self,
                                   for_train=True)

    def get_index(self, result, patch, position, index_use):
        # 该方案有点效率低下，主要因为level_area_location不能转化为area_location, 需要area_location转level_area_location时返回坐标关系才能直接映射
        # 所以这里直接输入patch，运用list的index的方法寻找坐标
        for i in range(len(result)):
            [slide_index, one_or_zero] = position[i]
            index = self.all_patch[index_use][slide_index][0][int(1 - one_or_zero)].index(patch[i])
            self.result_record[index_use][slide_index][int(1 - one_or_zero)][index].append(result[i])

    def record_result_mil(self, result, patch, position, index_use):
        # 该方案有点效率低下，主要因为level_area_location不能转化为area_location, 需要area_location转level_area_location时返回坐标关系才能直接映射
        # 所以这里直接输入patch，运用list的index的方法寻找坐标
        for i in range(len(result)):
            [slide_index, one_or_zero] = position[i]
            index = self.all_patch[index_use][slide_index][0][int(1 - one_or_zero)].index(patch[i])
            try:
                self.slide_record[index_use][slide_index][int(1 - one_or_zero)][index][0] = result[i]
            except:
                self.slide_record[index_use][slide_index][int(1 - one_or_zero)][index].append(result[i][0])

    def calc_result(self, index_use):
        result_label = []
        result_mil = []
        for i in range(len(self.information['use_list'][index_use])):
            j = self.information['use_list'][index_use][i]
            result_max_slide = self.result_max[index_use][i]
            label_mil = int(self.information['label'][j])
            result_use = result_max_slide[0][2]
            result_label.append(label_mil)
            result_mil.append(float(result_use))
        return result_mil, result_label


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
        slide_name, name, one_index_result, zero_index_result, one_level_result, zero_level_result, i, j = \
            self.all_result[position]
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


class DatasetForLoaderMIL:
    def __init__(self, index_use, result, one_num, zero_num, one_num_list, zero_num_list, init_dataset: InitDataset,
                 for_train=False):
        self.class_num = index_use
        self.all_result = result
        self.one_num = one_num
        self.zero_num = zero_num
        self.one_num_list = one_num_list
        self.zero_num_list = zero_num_list
        self.for_train = for_train
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
        label_mil = None
        one_label_mil = None
        if self.for_train is False:
            slide_name, name, one_index_result, zero_index_result, one_level_result, zero_level_result, i, j = \
                self.all_result[position]
        else:
            slide_name, name, one_index_result, zero_index_result, one_level_result, zero_level_result, i, j, one_label_mil, zero_label_mil = \
                self.all_result[position]
        slide = self.base_dataset.operate.read_slide(slide_name)
        index_position, index_index = self.get_index_in_index_result(index, one_or_zero, one_index_result,
                                                                     zero_index_result)
        if one_or_zero == 1:
            true_index = one_index_result[index_position][index_index]
            patch = one_level_result[index_position][true_index]
            if self.for_train is True:
                label_mil = one_label_mil[index_position][index_index]
        else:
            true_index = zero_index_result[index_position][index_index]
            patch = zero_level_result[index_position][true_index]
            if self.for_train is True:
                label_mil = zero_label_mil[index_position][index_index]
        [_, _, x, y, level, patch_size] = patch
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
        img = img / 255.0

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
        img_mil = img_flat[(len(label_flat) - 1) // 2]
        if self.for_train is False:
            label_mil = np.array([np.float32(self.base_dataset.information['label'][j])])
        return img_flat, label_flat, img_mil, label_mil, np.array(patch), np.array([position, one_or_zero])

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
                add_list.append(len(list_a[i]))
            else:
                add_list.append(len(list_a[i]) + add_list[i - 1])
        return add_list

    def __len__(self):
        return self.length
