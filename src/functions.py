###############################
#
# Created by Patrik Valkovic
# 3/2/2021
#
###############################

import torch as t
from .Problem import Problem
from . import utils

@utils.seedable
def create_f01(dim, dev=None) -> Problem:
    x_opt = utils.rand_xopt(dim, dev)
    f_opt = utils.rand_fopt(dev)
    def _f(x, x_opt, f_opt):
        z = x - x_opt[None, :]
        norm = t.sum(z * z, dim=-1)
        return norm + f_opt
    return Problem(
        _f, [x_opt, f_opt], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )


@utils.seedable
def create_f02(dim, dev=None) -> Problem:
    x_opt = utils.rand_xopt(dim, dev)
    f_opt = utils.rand_fopt(dev)
    mult = 6 * t.arange(0, dim, dtype=t.float32, device=dev) / (dim - 1)
    mult = t.pow(10, mult, out=mult)
    def _f(x, x_opt, f_opt, mult):
        z = utils.T_osz(x - x_opt[None, :])
        norm = t.sum(z * z * mult[None, :], dim=-1)
        return norm + f_opt
    return Problem(
        _f, [x_opt, f_opt, mult], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )


@utils.seedable
def create_f03(dim, dev=None) -> Problem:
    x_opt = utils.rand_xopt(dim, dev)
    f_opt = utils.rand_fopt(dev)
    lamb = utils.Lambda(10, dim, t.float32, dev)
    def _f(x, x_opt, f_opt, lamb):
        z_tmp = utils.T_asy(
            utils.T_osz(x - x_opt),
            0.2,
            dim,
        )
        z = (lamb @ z_tmp.T).T
        first_part = 10 * (dim - t.sum(t.cos(2*utils.PI*z), dim=-1))
        z_norm = t.sum(z * z, dim=-1)
        return first_part + z_norm + f_opt
    return Problem(
        _f, [x_opt, f_opt, lamb], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )


@utils.seedable
def create_f04(dim, dev=None) -> Problem:
    x_opt = utils.rand_xopt(dim, dev)
    f_opt = utils.rand_fopt(dev)
    s = t.pow(10, 0.5 * t.arange(0, dim, dtype=t.float32, device=dev) / (dim - 1))
    odd_indices = t.arange(0, dim, 2, dtype=t.long, device=dev)
    def _f(x, x_opt, f_opt, s, odd_indices):
        z = s[None, :] * utils.T_osz(x - x_opt)
        z_odd = z[:,odd_indices]
        z_odd[z_odd > 0] *= 10
        z[:,odd_indices] = z_odd
        first_part = 10 * (dim - t.sum(t.cos(2*utils.PI*z), dim=-1))
        second_part = t.sum(z * z, dim=-1)
        third_part = 100 * utils.f_pen(x)
        fourth_part = f_opt
        return first_part + second_part + third_part + fourth_part
    return Problem(
        _f, [x_opt, f_opt, s, odd_indices], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )


@utils.seedable
def create_f05(dim, dev=None) -> Problem:
    x_opt = 5 * (t.bernoulli(t.full((dim,), 0.5, dtype=t.float32, device=dev)) * 2 - 1)
    f_opt = utils.rand_fopt(dev)
    s = t.sign(x_opt) * t.pow(10, t.arange(0, dim, dtype=t.float32, device=dev) / (dim - 1))
    def _f(x, x_opt, f_opt, s):
        z = x_opt.repeat(x.shape[0]).reshape(x.shape)
        selector = x * x_opt[None, :] < 5 ** 2
        z[selector] = x[selector]
        f = t.sum(5 * t.abs(s[None, :]) -s[None, :] * z,dim = -1)
        return f + f_opt
    return Problem(
        _f, [x_opt, f_opt, s], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )