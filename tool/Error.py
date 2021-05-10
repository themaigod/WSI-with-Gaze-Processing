class RegisterError(Exception):
    def __init__(self, name=None):
        super(RegisterError, self).__init__()
        self.msg = name

    def __str__(self):
        return "This function " + self.msg + " is not registered"


class ModeError(Exception):
    def __init__(self, name=None):
        super(ModeError, self).__init__()
        self.msg = name

    def __str__(self):
        return "This mode " + self.msg + " is not exist"


class TypeError(Exception):
    def __init__(self, name=None):
        super(TypeError, self).__init__()
        self.msg = name

    def __str__(self):
        return "This type " + self.msg + " is not exist"


class ExistError(Exception):
    def __init__(self, name=None):
        super(ExistError, self).__init__()
        self.msg = name

    def __str__(self):
        return self.msg + " is not exist"


class OutBoundError(Exception):
    def __init__(self, name=None):
        super(OutBoundError, self).__init__()
        self.msg = name

    def __str__(self):
        return self.msg + " is out of bound"


class NoWriteError(Exception):
    def __init__(self, name=None):
        super().__init__()
        self.msg = name

    def __str__(self):
        return "base class" + self.msg + " is not write"
