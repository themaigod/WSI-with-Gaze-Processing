
from struct import *


class RECreader:
    def __init__(self, rec_file: str, erc_file: str, new_version=False):
        """
            读取REC和ERC文件。如果是新版ERC文件则要令newVersion=True
        """
        self.__new_version = new_version
        br = open(rec_file, "rb")
        ebr = open(erc_file, "rb")
        # 先读取rec文件尾
        br.seek(-4, 2)
        length = unpack("i", br.read(4))[0]  # 注释信息总长度
        br.seek(-length, 2)
        self.__comment = read_str(br)
        self.__type = read_str(br)
        # 从rec文件头读取医生信息
        br.seek(0, 0)
        is_man = unpack("?", br.read(1))[0]
        age = unpack("H", br.read(2))[0]
        time = unpack("H", br.read(2))[0]
        level = unpack("H", br.read(2))[0]
        self.__doctor = [is_man, age, time, level]
        self.__fileName = read_str(br)
        # 从erc文件未读取一个整数，代表记录的总帧数
        ebr.seek(-4, 2)
        self.__tickNum = unpack("i", ebr.read(4))[0]
        # 从erc文件头读取两个整数，代表录制时的屏幕宽高
        ebr.seek(0, 0)
        width = unpack("i", ebr.read(4))[0]
        height = unpack("i", ebr.read(4))[0]
        mm_height = unpack("f", ebr.read(4))[0]
        mm_width = unpack("f", ebr.read(4))[0]
        self.__screen = [width, height, mm_width, mm_height]
        self.__data = []
        now_tick = 0
        level = 8
        while True:
            if now_tick >= self.__tickNum:
                break
            # 获取模式信息
            mode = unpack("i", br.read(4))[0]
            if mode == 0:  # 移动
                sx = -int(unpack("f", br.read(4))[0] - width / 2) * level
                sy = int(unpack("f", br.read(4))[0] + height / 2) * level
                now_tick += 1
                x = unpack("i", ebr.read(4))[0]
                y = unpack("i", ebr.read(4))[0]
                z = unpack("f", ebr.read(4))[0]
                t = unpack("f", ebr.read(4))[0]
                if z > 0:
                    v = True
                else:
                    v = False
                self.__data.append([x, y, z, level, v, sx, sy, t])
            elif mode == 1:  # 放大
                if level > 1:
                    level /= 2
            elif mode == 2:  # 缩小
                if level < 8:
                    level *= 2
            elif mode == 3:  # 终止
                break
            else:  # 文件损坏
                raise Exception("Unexpected mode " + str(mode) + " at Tick " + str(now_tick))

    def get_doctor_info(self):
        return self.__doctor

    def get_slide_file_name(self):
        return self.__fileName

    def get_screen_info(self):
        return self.__screen

    def get_tick_num(self):
        return self.__tickNum

    def get_data(self):
        return self.__data

    def get_comment(self):
        return self.__comment

    def get_type(self):
        return self.__type


def read_str(reader):
    # 获取第一个长度前缀
    str_len = unpack("b", reader.read(1))[0]
    if str_len == 0:
        return ""
    # 判断是否有下一个长度前缀
    if str_len >> 7 == -1:
        str_len = str_len & 0b01111111 + unpack("b", reader.read(1))[0] * 128
    else:
        str_len = str_len & 0b01111111
    return str(reader.read(str_len), "utf-8")
