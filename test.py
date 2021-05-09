from produce_dataset import *


# def static_all_list(self, level_dimensions, detect_level, level_downsamples, patch_size, mode=0, location_type=0):
#     # 根据detect_level提供的level获得在这些level上的patch，包括detect location[location_type = 0]、area location[location_type = 1]两种形式，注意area location
#     # 形式的时候，注视点数量设为0。location_type = 2时提供一种不完备的形式, 即在代表每个patch元素中，同时储存detect location和area location
#     # 该location_type输出可以由static_separate_all_list_location_type_2转成正常的detect location和area location\\该方案已废弃
#     # 更好的解决方案可能是生成area location，再经由static_area_location2detect_location生成detect location
#     # mode指导返回数据
#     # mode=0，返回的形式是[list1, list2, list3...], list1是level=detect_level[0]时所有的patch，以此类推
#     # mode=1，返回标准的location，如，[location1, location2, ...], 包含所有的patch
#     # mode=2, 同时返回mode=0和mode=1的结果
#     result = []
#     result_optional = []
#     for i in range(len(detect_level)):
#         point = [level_dimensions[detect_level[i]][0] - 1, level_dimensions[detect_level[i]][0] - 1]
#         point_now_level = self.static_double_mul_div_int_mul_level(point, level_downsamples, detect_level[i], 0)
#         point_level_patch = self.static_mul_div(point_now_level, div=patch_size, as_int=True)
#         point_now_level_patch = self.static_mul_div(point_level_patch, mul=patch_size)
#         point_now_level_end = [0, 0]
#         if (point_now_level_patch[0] + patch_size - 1) != point_now_level:
#             point_now_level_end[0] = point_now_level_patch[0] - patch_size
#         else:
#             point_now_level_end[0] = point_now_level_patch[0]
#         if (point_now_level_patch[1] + patch_size - 1) != point_now_level:
#             point_now_level_end[1] = point_now_level_patch[1] - patch_size
#         else:
#             point_now_level_end[1] = point_now_level_patch[1]
#         loop = self.static_mul_div(point_now_level_end, div=patch_size, as_int=True)
#         result_level = []
#         for j in range(loop[0]):
#             for k in range(loop[1]):
#                 point_dynamic = self.static_transform_pixel([j, k], level_downsamples, patch_size, detect_level[i],
#                                                             0)
#                 if location_type == 0:
#                     location = [0, point_dynamic[0], point_dynamic[1], detect_level[i], patch_size]
#                 elif location_type == 1:
#                     location = [point_dynamic[0], point_dynamic[1], detect_level[i], patch_size]
#                 elif location_type == 2:
#                     location = [[0, point_dynamic[0], point_dynamic[1], detect_level[i], patch_size],
#                                 [point_dynamic[0], point_dynamic[1], detect_level[i], patch_size]]
#                 else:
#                     raise ModeError("location_type")
#                 if mode == 0:
#                     result_level.append(location)
#                 elif mode == 1:
#                     result.append(location)
#                 elif mode == 2:
#                     result_level.append(location)
#                     result.append(location)
#                 else:
#                     raise ModeError("in static_all_list")
#         if mode == 0:
#             result.append(result_level)
#         elif mode == 2:
#             result_optional.append(result_level)
#     if mode == 0 or mode == 1:
#         return result
#     elif mode == 2:
#         return result, result_optional
#     else:
#         raise ModeError("in static_all_list")
#
#
# self = DatasetRegularProcess(record=False)
#
# static_all_list(self, level_dimensions, detect_level, level_downsamples, patch_size, location_type=1)

[name, point_array, level_array, path, max_num, image_label] = all_reader(r"D:\ajmq\point_cs")
