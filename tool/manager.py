from .Error import (RegisterError, ModeError, ExistError, OutBoundError)
from .ClassSave import SingleClass


class Manager:  # 标准管理器
    def __init__(self, instance):
        self.instance = instance
        self.register_inner_func()

    def register_inner_func(self):  # 类内注册
        self.register(self.__init__)
        self.register(self.register_inner_func)
        self.register(self.register)
        self.register(self.record)
        self.register(self.func_run)
        self.register(self.inner_func_run)
        self.register(self.func_show)
        self.register(self.func_del)
        self.record("register_inner_func")

    def register(self, func_name, func_type=0, input_mode=0):  # 注册
        if func_name not in self.instance.func_name:
            self.instance.register(func_name, func_type, input_mode)
        self.record("register")

    def record(self, func_name, record=True):  # 记录
        if record is True:
            if func_name is not None:
                order = ["Manager", func_name]
            else:
                order = ["Manager"]
            self.instance.order.append(order)

    def func_run(self, func_name, inputs, func_type=0, input_mode=0):  # 运行函数，详情参见GetDataset里的介绍
        if func_name not in self.instance.func_name:
            raise RegisterError(func_name)
        if 'func_run' in dir(self.instance):
            result = self.instance.func_run(func_name, inputs, func_type, input_mode)
        else:
            if func_type == 0:
                pass
            elif func_type == 1:
                func_name = eval(func_name)
            else:
                raise ModeError("func_type {}".format(func_type))
            if input_mode == 0:
                result = func_name(*inputs)
                self.record(str(func_name))
            elif input_mode == 1:
                result = func_name(**inputs)
                self.record(str(func_name))
            else:
                raise ModeError("input_mode {}".format(input_mode))
        self.record("func_run")
        return result

    def inner_func_run(self, func_point, inputs, access_mode=0):  # 使用已被注册的函数在类内运行，并使用注册时的运行设置
        if access_mode == 0:
            index = self.instance.func_name.index(func_point)
            func_name = func_point
            func_type = self.instance.func_setting[index][0]
            input_mode = self.instance.func_setting[index][1]
        elif access_mode == 1:
            index = func_point
            func_name = self.instance.func_name[index]
            func_type = self.instance.func_setting[index][0]
            input_mode = self.instance.func_setting[index][1]
        else:
            raise ModeError("access_mode {}".format(access_mode))
        result = self.func_run(func_name, inputs, func_type, input_mode)
        self.record("inner_func_run")
        return result

    def func_show(self):  # 展示已注册的函数
        for i in range(len(self.instance.func_name)):
            print(self.instance.func_name[i])
        self.record("func_show")

    def func_del(self, func_point, access_mode):  # 销毁已注册函数
        if access_mode == 0:
            if func_point not in self.instance.func_name:
                raise ExistError("func_name: " + str(func_point))
            index = self.instance.func_name.index(func_point)
            self.instance.func_name.pop(index)
            self.instance.func_setting.pop(index)
        elif access_mode == 1:
            if func_point >= len(self.instance.func_name):
                raise OutBoundError("func_index: " + str(func_point))
            index_all = func_point
            self.instance.func_name.pop(index_all)
            self.instance.func_setting.pop(index_all)
        else:
            raise ModeError("access_mode {}".format(access_mode))
        self.record("func_del")


class Static(Manager):  # 对static属性的函数进行管理
    def __init__(self, instance):
        super().__init__(instance)

    def register(self, func_name, func_type=0, input_mode=0):
        if func_name not in self.instance.static_func_name:
            self.instance.static_func_name.append(func_name)
            self.instance.static_func_setting.append([func_type, input_mode])
            self.instance.register(func_name, func_type, input_mode)
        self.record("register")

    def record(self, func_name, record=True):
        if record is True:
            if func_name is not None:
                order = ["static", func_name]
            else:
                order = ["static"]
            self.instance.order.append(order)

    def func_run(self, func_name, inputs, func_type=0, input_mode=0):
        if func_name not in self.instance.static_func_name:
            raise RegisterError(func_name)
        if 'func_run' in dir(self.instance):
            result = self.instance.func_run(func_name, inputs, func_type, input_mode)
        else:
            if func_type == 0:
                pass
            elif func_type == 1:
                func_name = eval(func_name)
            else:
                raise ModeError("func_type {}".format(func_type))
            if input_mode == 0:
                result = func_name(*inputs)
                self.record(str(func_name))
            elif input_mode == 1:
                result = func_name(**inputs)
                self.record(str(func_name))
            else:
                raise ModeError("input_mode {}".format(input_mode))
        self.record("func_run")
        return result

    def inner_func_run(self, func_point, inputs, access_mode=0):
        if access_mode == 0:
            index = self.instance.static_func_name.index(func_point)
            func_name = func_point
            func_type = self.instance.static_func_setting[index][0]
            input_mode = self.instance.static_func_setting[index][1]
        elif access_mode == 1:
            index = func_point
            func_name = self.instance.static_func_name[index]
            func_type = self.instance.static_func_setting[index][0]
            input_mode = self.instance.static_func_setting[index][1]
        else:
            raise ModeError("access_mode {}".format(access_mode))
        result = self.func_run(func_name, inputs, func_type, input_mode)
        self.record("inner_func_run")
        return result

    def func_show(self):
        for i in range(len(self.instance.static_func_name)):
            print(self.instance.static_func_name[i])
        self.record("func_show")

    def func_del(self, func_point, access_mode):
        if access_mode == 0:
            if func_point not in self.instance.static_func_name:
                raise ExistError("func_name: " + str(func_point))
            index = self.instance.static_func_name.index(func_point)
            self.instance.func_name.pop(index)
            self.instance.func_setting.pop(index)
            index_all = self.instance.func_name.index(func_point)
            self.instance.func_name.pop(index_all)
            self.instance.func_setting.pop(index_all)
        elif access_mode == 1:
            if func_point >= len(self.instance.static_func_name):
                raise OutBoundError("func_index: " + str(func_point))
            index = func_point
            func_name = self.instance.static_func_name[index]
            self.instance.static_func_name.pop(index)
            self.instance.static_func_setting.pop(index)
            index_all = self.instance.func_name.index(func_name)
            self.instance.func_name.pop(index_all)
            self.instance.func_setting.pop(index_all)
        else:
            raise ModeError("access_mode {}".format(access_mode))
        self.record("func_del")


class Inner(Manager):  # 对inner属性函数进行管理
    def __init__(self, instance):
        super().__init__(instance)

    def register(self, func_name, func_type=0, input_mode=0):
        if func_name not in self.instance.inner_func_name:
            self.instance.inner_func_name.append(func_name)
            self.instance.inner_func_setting.append([func_type, input_mode])
            self.instance.register(func_name, func_type, input_mode)
        self.record("register")

    def record(self, func_name, record=True):
        if record is True:
            if func_name is not None:
                order = ["inner", func_name]
            else:
                order = ["inner"]
            self.instance.order.append(order)

    def func_run(self, func_name, inputs, func_type=0, input_mode=0):
        if func_name not in self.instance.inner_func_name:
            raise RegisterError(func_name)
        if 'func_run' in dir(self.instance):
            result = self.instance.func_run(func_name, inputs, func_type, input_mode)
        else:
            if func_type == 0:
                pass
            elif func_type == 1:
                func_name = eval(func_name)
            else:
                raise ModeError("func_type {}".format(func_type))
            if input_mode == 0:
                result = func_name(*inputs)
                self.record(str(func_name))
            elif input_mode == 1:
                result = func_name(**inputs)
                self.record(str(func_name))
            else:
                raise ModeError("input_mode {}".format(input_mode))
        self.record("func_run")
        return result

    def inner_func_run(self, func_point, inputs, access_mode=0):
        if access_mode == 0:
            index = self.instance.inner_func_name.index(func_point)
            func_name = func_point
            func_type = self.instance.inner_func_setting[index][0]
            input_mode = self.instance.inner_func_setting[index][1]
        elif access_mode == 1:
            index = func_point
            func_name = self.instance.inner_func_name[index]
            func_type = self.instance.inner_func_setting[index][0]
            input_mode = self.instance.inner_func_setting[index][1]
        else:
            raise ModeError("access_mode {}".format(access_mode))
        result = self.func_run(func_name, inputs, func_type, input_mode)
        self.record("inner_func_run")
        return result

    def func_show(self):
        for i in range(len(self.instance.inner_func_name)):
            print(self.instance.inner_func_name[i])
        self.record("func_show")

    def func_del(self, func_point, access_mode):
        if access_mode == 0:
            if func_point not in self.instance.inner_func_name:
                raise ExistError("func_name: " + str(func_point))
            index = self.instance.inner_func_name.index(func_point)
            self.instance.func_name.pop(index)
            self.instance.func_setting.pop(index)
            index_all = self.instance.func_name.index(func_point)
            self.instance.func_name.pop(index_all)
            self.instance.func_setting.pop(index_all)
        elif access_mode == 1:
            if func_point >= len(self.instance.inner_func_name):
                raise OutBoundError("func_index: " + str(func_point))
            index = func_point
            func_name = self.instance.inner_func_name[index]
            self.instance.inner_func_name.pop(index)
            self.instance.inner_func_setting.pop(index)
            index_all = self.instance.func_name.index(func_name)
            self.instance.func_name.pop(index_all)
            self.instance.func_setting.pop(index_all)
        else:
            raise ModeError("access_mode {}".format(access_mode))
        self.record("func_del")


class SingleManager(Manager):
    # 如果有特殊想拆分开来的类别的函数，可使用该管理器，该类的manager_list初始化ClassSave.py里的SingleClass，将函数名和设置存于该对象
    def __init__(self, instance, manager_name):
        self.manager_list = SingleClass()
        self.manager_name = manager_name
        super().__init__(instance)

    def register(self, func_name, func_type=0, input_mode=0):
        if func_name not in self.manager_list.func_name:
            self.manager_list.func_name.append(func_name)
            self.manager_list.func_setting.append([func_type, input_mode])
            self.instance.register(func_name, func_type, input_mode)
        self.record("register")

    def record(self, func_name, record=True):
        if record is True:
            if func_name is not None:
                order = [self.manager_name, func_name]
            else:
                order = [self.manager_name]
            self.instance.order.append(order)

    def func_run(self, func_name, inputs, func_type=0, input_mode=0):
        if 'func_run' in dir(self.instance):
            result = self.instance.func_run(func_name, inputs, func_type, input_mode)
        else:
            if func_name not in self.manager_list.func_name:
                raise RegisterError(func_name)
            if func_type == 0:
                pass
            elif func_type == 1:
                func_name = eval(func_name)
            else:
                raise ModeError("func_type {}".format(func_type))
            if input_mode == 0:
                result = func_name(*inputs)
                self.record(str(func_name))
            elif input_mode == 1:
                result = func_name(**inputs)
                self.record(str(func_name))
            else:
                raise ModeError("input_mode {}".format(input_mode))
        self.record("func_run")
        return result

    def inter_func_run(self, func_point, inputs, access_mode=0):
        if access_mode == 0:
            index = self.manager_list.func_name.index(func_point)
            func_name = func_point
            func_type = self.manager_list.func_setting[index][0]
            input_mode = self.manager_list.func_setting[index][1]
        elif access_mode == 1:
            index = func_point
            func_name = self.manager_list.func_name[index]
            func_type = self.manager_list.func_setting[index][0]
            input_mode = self.manager_list.func_setting[index][1]
        else:
            raise ModeError("access_mode {}".format(access_mode))
        result = self.func_run(func_name, inputs, func_type, input_mode)
        self.record("inter_func_run")
        return result

    def func_show(self):
        for i in range(len(self.manager_list.func_name)):
            print(self.manager_list.func_name[i])
        self.record("func_show")

    def func_del(self, func_point, access_mode):
        if access_mode == 0:
            if func_point not in self.manager_list.func_name:
                raise ExistError("func_name: " + str(func_point))
            index = self.manager_list.func_name.index(func_point)
            self.instance.func_name.pop(index)
            self.instance.func_setting.pop(index)
            index_all = self.instance.func_name.index(func_point)
            self.instance.func_name.pop(index_all)
            self.instance.func_setting.pop(index_all)
        elif access_mode == 1:
            if func_point >= len(self.manager_list.func_name):
                raise OutBoundError("func_index: " + str(func_point))
            index = func_point
            func_name = self.manager_list.func_name[index]
            self.manager_list.func_name.pop(index)
            self.manager_list.func_setting.pop(index)
            index_all = self.instance.func_name.index(func_name)
            self.instance.func_name.pop(index_all)
            self.instance.func_setting.pop(index_all)
        else:
            raise ModeError("access_mode {}".format(access_mode))
        self.record("func_del")
