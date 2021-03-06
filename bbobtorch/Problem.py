###############################
#
# Created by Patrik Valkovic
# 3/2/2021
#
###############################

class Problem:
    def __init__(self, function, parameters, x_opt, f_opt, x_min, x_max):
        self._function = function
        self._parameters = parameters
        self._dimension = x_opt.shape[0]
        self._x_opt = x_opt
        self._f_opt = f_opt
        self._x_min = x_min
        self._x_max = x_max

    def __call__(self, x):
        return self._function(x, *self._parameters)

    @property
    def dim(self):
        return self._dimension

    @property
    def x_opt(self):
        return self._x_opt

    @property
    def f_opt(self):
        return self._f_opt

    def to(self, *args, **kargs):
        for i in range(len(self._parameters)):
            self._parameters[i] = self._parameters[i].to(*args, **kargs)
        self._x_opt = self._x_opt.to(*args, **kargs)
        self._f_opt = self._f_opt.to(*args, **kargs)
        self._x_min = self._x_min.to(*args, **kargs)
        self._x_max = self._x_max.to(*args, **kargs)


    def type(self, dtype, *args, **kargs):
        for i in range(len(self._parameters)):
            self._parameters[i] = self._parameters[i].type(dtype, *args, **kargs)
        self._x_opt = self._x_opt.type(dtype, *args, **kargs)
        self._f_opt = self._f_opt.type(dtype, *args, **kargs)
        self._x_min = self._x_min.type(dtype, *args, **kargs)
        self._x_max = self._x_max.type(dtype, *args, **kargs)

    @property
    def dtype(self):
        return self._parameters[0].dtype

    @property
    def x_min(self):
        return self._x_min

    @property
    def x_max(self):
        return self._x_max
