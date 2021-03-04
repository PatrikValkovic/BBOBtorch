###############################
#
# Created by Patrik Valkovic
# 3/2/2021
#
###############################

import math
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


# TODO is correct?
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

@utils.seedable
def create_f06(dim, dev=None) -> Problem:
    x_opt = utils.rand_xopt(dim, dev)
    f_opt = utils.rand_fopt(dev)
    R = utils.rotation_matrix(dim, t.float32, dev)
    Q = utils.rotation_matrix(dim, t.float32, dev)
    lamb = utils.Lambda(10, dim, t.float32, dev)
    def _f(x, x_opt, f_opt, R, Q, lamb):
        z = (Q @ lamb @ R @ (x - x_opt).T).T
        s = t.ones_like(x, dtype=x.dtype, device=x.device)
        s[z * x_opt[None,:] > 0] = 100
        f = t.pow(utils.T_osz(t.sum(t.pow(s * z, 2), dim=-1)), 0.9)
        return f + f_opt
    return Problem(
        _f, [x_opt, f_opt, R, Q, lamb], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )


@utils.seedable
def create_f07(dim, dev=None) -> Problem:
    x_opt = utils.rand_xopt(dim, dev)
    f_opt = utils.rand_fopt(dev)
    R = utils.rotation_matrix(dim, t.float32, dev)
    Q = utils.rotation_matrix(dim, t.float32, dev)
    lamb = utils.Lambda(10, dim, t.float32, dev)
    mult = t.pow(10, 2 * t.arange(0, dim, dtype=t.float32, device=dev) / (dim - 1))
    def _f(x, x_opt, f_opt, R, Q, lamb, mult):
        z_hat = (lamb @ R @ (x - x_opt).T).T
        z_dash = t.full_like(x, 0.5)
        greater = t.abs(z_hat) > 0.5
        z_dash[greater] += z_hat[greater]
        lower = t.logical_not(greater, out=greater)
        z_dash[lower] += 10 * z_hat[lower]
        z_dash = t.floor_(z_dash)
        z_dash[lower] /= 10
        z = (Q @ z_dash.T).T
        f = 0.1 * t.maximum(
            t.abs(z_hat[:,0]) / 10**4,
            t.sum(mult[None, :] * z * z, dim=-1)
        )
        return f + utils.f_pen(x) + f_opt
    return Problem(
        _f, [x_opt, f_opt, R, Q, lamb, mult], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )


@utils.seedable
def create_f08(dim, dev=None) -> Problem:
    x_opt = utils.rand_xopt(dim, dev) / 5.0 * 3.0
    f_opt = utils.rand_fopt(dev)
    zmax = max(1.0, math.sqrt(dim) / 8.0)
    def _f(x, x_opt, f_opt, zmax):
        z = zmax * (x - x_opt[None,:]) + 1
        unshift = t.index_select(z, dim=-1, index=t.arange(0, dim-1, device=x.device, dtype=t.long))
        shifted = t.index_select(z, dim=-1, index=t.arange(1, dim, device=x.device, dtype=t.long))
        s = t.sum(
            100 * t.pow(t.pow(unshift, 2) - shifted,2) + t.pow(unshift - 1, 2),
            dim=-1
        )
        return s + f_opt
    return Problem(
        _f, [x_opt, f_opt, zmax], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )


@utils.seedable
def create_f09(dim, dev=None) -> Problem:
    f_opt = utils.rand_fopt(dev)
    R = utils.rotation_matrix(dim, dtype=t.float32, dev=dev)
    zmax = max(1.0, math.sqrt(dim) / 8.0)
    x_opt = t.ones(size=(dim,), dtype=t.float32, device=dev)
    x_opt = (x_opt - 0.5) / zmax
    Rinv = t.inverse(R)
    x_opt = Rinv @ x_opt
    def _f(x, f_opt, R, zmax):
        z = zmax * (R @ x.T).T + 0.5
        unshift = t.index_select(z, dim=-1, index=t.arange(0, dim-1, device=x.device, dtype=t.long))
        shifted = t.index_select(z, dim=-1, index=t.arange(1, dim, device=x.device, dtype=t.long))
        s = t.sum(
            100 * t.pow(t.pow(unshift, 2) - shifted,2) + t.pow(unshift - 1, 2),
            dim=-1
        )
        return s + f_opt
    return Problem(
        _f, [f_opt, R, zmax], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )


@utils.seedable
def create_f10(dim, dev=None) -> Problem:
    f_opt = utils.rand_fopt(dev)
    x_opt = utils.rand_xopt(dim, dev)
    mult = t.pow(10, 6 * t.arange(0, dim, dtype=t.float32, device=dev) / (dim - 1))
    R = utils.rotation_matrix(dim, t.float32, dev).T
    def _f(x, f_opt, x_opt, mult, R):
        z = utils.T_osz((x - x_opt[None,:]) @ R)
        f = t.sum(mult[None,:] * t.pow(z, 2), dim=-1)
        return f + f_opt
    return Problem(
        _f, [f_opt, x_opt, mult, R], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )


@utils.seedable
def create_f11(dim, dev=None) -> Problem:
    f_opt = utils.rand_fopt(dev)
    x_opt = utils.rand_xopt(dim, dev)
    R = utils.rotation_matrix(dim, t.float32, dev).T
    def _f(x, f_opt, x_opt, R):
        z = utils.T_osz((x - x_opt[None,:]) @ R)
        first = 10**6 * t.pow(z[:,0], 2)
        second = t.sum(t.pow(z[:,1:], 2), dim=-1)
        return first + second + f_opt
    return Problem(
        _f, [f_opt, x_opt, R], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )

# TODO is correct?
@utils.seedable
def create_f12(dim, dev=None) -> Problem:
    f_opt = utils.rand_fopt(dev)
    x_opt = utils.rand_xopt(dim, dev)
    R = utils.rotation_matrix(dim, t.float32, dev).T
    def _f(x, f_opt, x_opt, R):
        z = utils.T_asy((x - x_opt[None, :]) @ R, 0.5, dim) @ R
        first = t.pow(z[:,0], 2)
        second = 10 ** 6 * t.sum(t.pow(z[:,1:], 2), dim=-1)
        return first + second + f_opt
    return Problem(
        _f, [f_opt, x_opt, R], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )


@utils.seedable
def create_f13(dim, dev=None) -> Problem:
    f_opt = utils.rand_fopt(dev)
    x_opt = utils.rand_xopt(dim, dev)
    R = utils.rotation_matrix(dim, t.float32, dev).T
    Q = utils.rotation_matrix(dim, t.float32, dev).T
    lamb = utils.Lambda(10, dim, t.float32, dev)
    def _f(x, f_opt, x_opt, R, Q, lamb):
        z = (x - x_opt[None, :]) @ R @ lamb @ Q
        first = t.pow(z[:,0], 2)
        second = 100 * t.sqrt(t.sum(t.pow(z[:,1:], 2), dim=-1))
        return first + second + f_opt
    return Problem(
        _f, [f_opt, x_opt, R, Q, lamb], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )


# TODO plot doesnt match
@utils.seedable
def create_f14(dim, dev=None) -> Problem:
    f_opt = utils.rand_fopt(dev)
    x_opt = utils.rand_xopt(dim, dev)
    R = utils.rotation_matrix(dim, t.float32, dev).T
    pow = 2 + 4 * t.arange(0, dim, dtype=t.float32, device=dev) / (dim - 1)
    def _f(x, f_opt, x_opt, R, pow):
        z = (x - x_opt[None, :]) @ R
        f = t.sqrt(t.sum(t.pow(t.abs(z), pow[None, :]), dim=-1))
        return f + f_opt
    return Problem(
        _f, [f_opt, x_opt, R, pow], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )


