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
    lamb = utils.Lambda(10, dim, t.float32, dev).T
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


@utils.seedable
def create_f15(dim, dev=None) -> Problem:
    f_opt = utils.rand_fopt(dev)
    x_opt = utils.rand_xopt(dim, dev)
    R = utils.rotation_matrix(dim, t.float32, dev).T
    Q = utils.rotation_matrix(dim, t.float32, dev).T
    lamb = utils.Lambda(10, dim, t.float32, dev).T
    def _f(x, f_opt, x_opt, R, Q, lamb):
        z = utils.T_asy(
            utils.T_osz((x - x_opt[None, :]) @ R),
            0.2, dim
        ) @ Q @ lamb @ R
        f = 10 * (dim - t.sum(t.cos(2 * utils.PI * z), dim=-1))
        s = t.sum(t.pow(z, 2), dim=-1)
        return f + s + f_opt
    return Problem(
        _f, [f_opt, x_opt, R, Q, lamb], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )


@utils.seedable
def create_f16(dim, dev=None) -> Problem:
    f_opt = utils.rand_fopt(dev)
    x_opt = utils.rand_xopt(dim, dev)
    R = utils.rotation_matrix(dim, t.float32, dev).T
    Q = utils.rotation_matrix(dim, t.float32, dev).T
    lamb = utils.Lambda(1.0/100.0, dim, t.float32, dev).T
    r11 = t.arange(0, 12)
    f_0 = t.sum(1 / t.pow(2, r11) * t.cos(2*utils.PI*t.pow(3, r11)*0.5))
    def _f(x, f_opt, x_opt, R, Q, lamb, f_0, r11):
        z = utils.T_osz(
            (x - x_opt[None, :]) @ R
        ) @ Q @ lamb @ R
        after_sums = t.pow(0.5, r11[:,None,None]) * t.cos(2*utils.PI*t.pow(3, r11[:,None,None])*(z[None,:,:] + 0.5))
        summed = 10 * t.pow(1/dim * t.sum(t.sum(after_sums, dim=0), dim=-1) - f_0,3)
        penalization = 10.0 / dim * utils.f_pen(x)
        return summed + penalization + f_opt
    return Problem(
        _f, [f_opt, x_opt, R, Q, lamb, f_0, r11], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )


@utils.seedable
def create_f17(dim, dev=None) -> Problem:
    f_opt = utils.rand_fopt(dev)
    x_opt = utils.rand_xopt(dim, dev)
    R = utils.rotation_matrix(dim, t.float32, dev).T
    Q = utils.rotation_matrix(dim, t.float32, dev).T
    lamb = utils.Lambda(1.0/100.0, dim, t.float32, dev).T
    def _f(x, f_opt, x_opt, R, Q, lamb):
        z = utils.T_asy(
            (x - x_opt[None,:]) @ R,
            0.5, dim
        ) @ Q @ lamb
        s = t.pow(z, 2, out=z)
        s = t.sqrt(s[:, :-1] + s[:, 1:])
        bracket = 1.0 / (dim - 1.0) * t.sum(
            t.sqrt(s) + t.sqrt(s) * t.pow(t.sin(50 * t.pow(s, 1.0 / 5.0)), 2),
            dim=-1
        )
        f = t.pow(bracket, 2, out=bracket)
        penalization = 10 * utils.f_pen(x)
        return f + penalization + f_opt
    return Problem(
        _f, [f_opt, x_opt, R, Q, lamb], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )



@utils.seedable
def create_f18(dim, dev=None) -> Problem:
    f_opt = utils.rand_fopt(dev)
    x_opt = utils.rand_xopt(dim, dev)
    R = utils.rotation_matrix(dim, t.float32, dev).T
    Q = utils.rotation_matrix(dim, t.float32, dev).T
    lamb = utils.Lambda(1000, dim, t.float32, dev).T
    def _f(x, f_opt, x_opt, R, Q, lamb):
        z = utils.T_asy(
            (x - x_opt[None,:]) @ R,
            0.5, dim
        ) @ Q @ lamb
        s = t.pow(z, 2, out=z)
        s = t.sqrt(s[:, :-1] + s[:, 1:])
        bracket = 1.0 / (dim - 1.0) * t.sum(
            t.sqrt(s) + t.sqrt(s) * t.pow(t.sin(50 * t.pow(s, 1.0 / 5.0)), 2),
            dim=-1
        )
        f = t.pow(bracket, 2, out=bracket)
        penalization = 10 * utils.f_pen(x)
        return f + penalization + f_opt
    return Problem(
        _f, [f_opt, x_opt, R, Q, lamb], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )


@utils.seedable
def create_f19(dim, dev=None) -> Problem:
    f_opt = utils.rand_fopt(dev)
    zmax = max(1.0, math.sqrt(dim) / 8.0)
    R = utils.rotation_matrix(dim, t.float32, dev).T
    x_opt = t.ones(size=(dim,), dtype=t.float32, device=dev)
    x_opt = (x_opt - 0.5) / zmax
    Rinv = t.inverse(R.T)
    x_opt = Rinv @ x_opt
    def _f(x, f_opt, R, zmax):
        z = zmax * x @ R + 0.5
        s = 100 * t.pow(t.pow(z[:,:-1], 2) - z[:,1:], 2) + t.pow(z[:,:-1] - 1, 2)
        f = 10.0 / (dim - 1) * t.sum(s / 4000 - t.cos(s), dim=-1)
        return f + 10 + f_opt
    return Problem(
        _f, [f_opt, R, zmax], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )


@utils.seedable
def create_f20(dim, dev=None) -> Problem:
    ones = t.bernoulli(t.full((dim,), 0.5, dtype=t.float32, device=dev)) * 2 - 1
    x_opt = 4.2096874633 / 2.0 * ones
    f_opt = utils.rand_fopt(dev)
    lamb = utils.Lambda(10, dim, t.float32, dev).T
    def _f(x, f_opt, x_opt, lamb, ones):
        x_hat = 2 * ones[None,:] * x
        z_hat = x_hat
        z_hat[:,1:] += 0.25 * (x_hat[:,:-1] - 2 * t.abs(x_opt[None,:-1]))
        z = 100 * ((z_hat - 2 * t.abs(x_opt[None, :])) @ lamb + 2 * t.abs(x_opt[None,:]))
        f = -1.0 / 100.0 / dim * t.sum(z * t.sin(t.sqrt(t.abs(z))), dim=-1)
        pen = 100 * utils.f_pen(z / 100)
        return f + 4.189828872724339 + pen + f_opt
    return Problem(
        _f, [f_opt, x_opt, lamb, ones], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )


@utils.seedable
def create_f21(dim, dev=None) -> Problem:
    optim = utils.random_optims(101, dim, 5.0, 4.0, dev)
    x_opt = optim[0]
    f_opt = utils.rand_fopt(dev)
    C = utils.Cmatrix(101, 100, dim, dev)
    w = utils.wvector(101, dev)
    R = utils.rotation_matrix(dim, t.float32, dev).T
    def _f(x, f_opt, y, C, w, R):
        diff = x[:,None,:] - y[None,:,:]
        multiplied_by_c = t.einsum("mwi,ijw->mwi", diff @ R, C)
        multiplied_by_c @= R.T
        in_bracket = t.multiply(multiplied_by_c, diff, out=diff)
        in_bracket = -1.0 / (2.0*dim) * t.sum(in_bracket, dim=-1)
        inside_max = t.multiply(w[None,:], t.exp(in_bracket))
        max, _ = t.max(inside_max, dim=-1)
        tosz = utils.T_osz(10 - max)
        tosz = t.pow(tosz, 2, out=tosz)
        pen = utils.f_pen(x)
        return tosz + pen + f_opt
    return Problem(
        _f, [f_opt, optim, C, w, R], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )


@utils.seedable
def create_f22(dim, dev=None) -> Problem:
    optim = utils.random_optims(21, dim, 4.9, 3.92, dev)
    x_opt = optim[0]
    f_opt = utils.rand_fopt(dev)
    C = utils.Cmatrix(21, 20, dim, dev, first_pow=2)
    w = utils.wvector(21, dev)
    R = utils.rotation_matrix(dim, t.float32, dev).T
    def _f(x, f_opt, y, C, w, R):
        diff = x[:, None, :] - y[None, :, :]
        multiplied_by_c = t.einsum("mwi,ijw->mwi", diff @ R, C)
        multiplied_by_c @= R.T
        in_bracket = t.multiply(multiplied_by_c, diff, out=diff)
        in_bracket = -1.0 / (2.0 * dim) * t.sum(in_bracket, dim=-1)
        inside_max = t.multiply(w[None, :], t.exp(in_bracket))
        max, _ = t.max(inside_max, dim=-1)
        tosz = utils.T_osz(10 - max)
        tosz = t.pow(tosz, 2, out=tosz)
        pen = utils.f_pen(x)
        return tosz + pen + f_opt
    return Problem(
        _f, [f_opt, optim, C, w, R], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )


@utils.seedable
def create_f23(dim, dev=None) -> Problem:
    x_opt = utils.rand_xopt(dim, dev)
    f_opt = utils.rand_fopt(dev)
    R = utils.rotation_matrix(dim, t.float32, dev).T
    Q = utils.rotation_matrix(dim, t.float32, dev).T
    lamb = utils.Lambda(100.0, dim, t.float32, dev).T
    J = 2 ** t.arange(1, 33, dtype=t.float32, device=dev)
    def _f(x, f_opt, x_opt, R, Q, lamb, J):
        z = (x - x_opt[None,:]) @ R @ lamb @ Q
        jsum = t.abs(J[None, :, None] * z[:, None, :] - t.round_(J[None, :, None]*z[:,None,:])) / J[None, :, None]
        bracket = t.sum(jsum, dim=1) * t.arange(1, dim+1, dtype=x.dtype, device=x.device)[None,:] + 1
        prod = t.prod(bracket, dim=-1)
        prod = t.pow(prod, 10/ dim ** 1.2, out=prod)
        prod *= 10.0 / dim ** 2.0
        prod -= 10 / dim ** 2.0
        prod += utils.f_pen(x)
        prod += f_opt
        return prod
    return Problem(
        _f, [f_opt, x_opt, R, Q, lamb, J], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )


@utils.seedable
def create_f24(dim, dev=None) -> Problem:
    s = 1 - 1/(2*math.sqrt(dim+20.0)-8.2)
    d = 1
    mu_0 = 2.5
    mu_1 = -math.sqrt((mu_0 ** 2 - d)/s)
    R = utils.rotation_matrix(dim, t.float32, dev).T
    Q = utils.rotation_matrix(dim, t.float32, dev).T
    lamb = utils.Lambda(100, dim, t.float32, dev)
    ones = t.ones(size=(dim,), dtype=t.float32, device=dev)
    f_opt = utils.rand_fopt()
    x_opt = mu_0 / 2.0 * t.bernoulli(t.full((dim,), 0.5, dtype=t.float32, device=dev)) * 2.0 - 1.0
    def _f(x, R, Q, lamb, ones, f_opt, x_opt):
        x_hat = 2 * t.sign(x_opt)[None,:] * x
        z = (x_hat - mu_0) @ R @ lamb @ Q
        min1 = t.sum(t.pow(x_hat - mu_0, 2), dim=-1)
        min2 = d*dim + s * t.sum(t.pow(x_hat - mu_1, 2), dim=-1)
        m = t.min(min1, min2, out=min1)
        second = 10 * (dim - t.sum(t.cos(2*utils.PI*z), dim=-1, out=min2))
        second += 10 ** 4 * utils.f_pen(x)
        second += f_opt
        second += m
        return second
    return Problem(
        _f, [R, Q, lamb, ones, f_opt, x_opt], x_opt, f_opt,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * -5,
        t.ones(size=(dim,), dtype=t.float32, device=dev) * 5
    )