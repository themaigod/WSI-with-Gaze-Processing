from produce_dataset import (GetInitDataset, DatasetRegularProcess)


class FullProcess(GetInitDataset, DatasetRegularProcess):
    def __init__(self, erc_path, record=True, func_mode=0, init_status=True, repeat_add_mode=True):
        super().__init__(erc_path, record, func_mode, init_status, repeat_add_mode)


if __name__ == '__main__':
    process_func = FullProcess(r"D:\ajmq\point_test", init_status=True)
    process_func.path = process_func.name2path_in_list(process_func.path, r"D:\ajmq\data_no_process")
    information, result, class_one_num, class_zero_num = process_func.process(process_func.name_array,
                                                                              process_func.path,
                                                                              process_func.point_array,
                                                                              process_func.level_array,
                                                                              None, process_func.image_label,
                                                                              process_func.max_num, process_func.config)
