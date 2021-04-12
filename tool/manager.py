from Error import (RegisterError, ModeError, ExistError, OutBoundError)


class Manager:
    def __init__(self, instance):
        self.instance.instance = instance
        self.register_inter_func()

    def register_inter_func(self):
        self.register(self.__init__)
        self.register(self.register_inter_func)
        self.register(self.register)
        self.register(self.record)
        self.register(self.func_run)
        self.register(self.inter_func_run)
        self.register(self.func_show)
        self.register(self.func_del)

    def register(self, func_name, func_type=0, input_mode=0):
        if func_name not in self.instance.func_name:
            self.instance.func_name.append(func_name)
            self.instance.func_setting.append([func_type, input_mode])
        order = ["Manager", "register"]
        self.instance.order.append(order)

    def record(self, func_name):
        if func_name is not None:
            order = ["Manager", func_name]
        else:
            order = ["Manager"]
        self.instance.order.append(order)

    def func_run(self, func_name, inputs, func_type=0, input_mode=0):
        if func_name not in self.instance.func_name:
            raise RegisterError(func_name)
        order_name = func_name
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
        order = ["Manager", order_name]
        self.instance.order.append(order)
        return result

    def inter_func_run(self, func_point, inputs, access_mode=0):
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
        self.func_run(func_name, inputs, func_type, input_mode)
        order = ["Manager", "inter_func_run"]
        self.instance.order.append(order)
        return result

    def func_show(self):
        for i in range(len(self.instance.func_name)):
            print(self.instance.func_name[i])
        order = ["Manager", "func_show"]
        self.instance.order.append(order)

    def func_del(self, func_point, access_mode):
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


class Static(Manager):
    def __init__(self, instance):
        super().__init__(instance)

    def register(self, func_name, func_type=0, input_mode=0):
        if func_name not in self.instance.static_func_name:
            self.instance.static_func_name.append(func_name)
            self.instance.static_func_setting.append([func_type, input_mode])
            self.instance.func_name.append(func_name)
            self.instance.func_setting.append([func_type, input_mode])
        order = ["static", "register"]
        self.instance.order.append(order)

    def record(self, func_name):
        if func_name is not None:
            order = ["static", func_name]
        else:
            order = ["static"]
        self.instance.order.append(order)

    def func_run(self, func_name, inputs, func_type=0, input_mode=0):
        if func_name not in self.instance.static_func_name:
            raise RegisterError(func_name)
        order_name = func_name
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
        order = ["static", order_name]
        self.instance.order.append(order)
        return result

    def inter_func_run(self, func_point, inputs, access_mode=0):
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
        self.instance.func_run(func_name, inputs, func_type, input_mode)
        order = ["static", "inter_func_run"]
        self.instance.order.append(order)
        return result

    def func_show(self):
        for i in range(len(self.instance.static_func_name)):
            print(self.instance.static_func_name[i])
        order = ["static", "func_show"]
        self.instance.order.append(order)

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
            func_name = self.static_func_name[index]
            self.static_func_name.pop(index)
            self.static_func_setting.pop(index)
            index_all = self.func_name.index(func_name)
            self.func_name.pop(index_all)
            self.func_setting.pop(index_all)
        else:
            raise ModeError("access_mode {}".format(access_mode))
