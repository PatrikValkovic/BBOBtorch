###############################
#
# Created by Patrik Valkovic
# 3/2/2021
#
###############################
import torch as t

PI = 3.15159265


def T_osz(x) -> t.Tensor:
    x_hat = t.log(t.abs(x))
    x_hat[t.isneginf(x_hat)] = 0
    c1 = t.full_like(x, 5.5)
    c1[x > 0] = 10.0
    c2 = t.full_like(x, 3.1)
    c2[x > 0] = 7.9
    return t.sign(x)*t.exp(x_hat + 0.049 * (t.sin(c1 * x_hat + t.sin(c2 * x_hat))))

def T_asy(x, beta, dim) -> t.Tensor:
    res = t.clone(x)
    pow = t.arange(0, dim, dtype=x.dtype, device=x.device)
    pow = 1 + beta * pow[None, :] / (dim - 1) * t.sqrt(x)
    pow = t.pow(x, pow, out=pow)
    res[res > 0] = pow[res > 0]
    return res

def Lambda(alpha, dim, dtype, dev) -> t.Tensor:
    diag = t.pow(
        alpha,
        0.5 * t.arange(0, dim, dtype=dtype, device=dev) / (dim - 1)
    )
    return t.diag(diag)

def f_pen(x) -> t.Tensor:
    return t.sum(
        t.pow(t.maximum(t.abs(x) - 5, t.zeros((1,), dtype=x.dtype, device=x.device)), 2),
        dim=-1
    )

def rotation_matrix(dim, dtype, dev) -> t.Tensor:
    R = t.randn(size=(dim,dim), dtype=dtype, device=dev)
    R, _ = t.qr(R)
    return R

def rand_xopt(dim, dev = None):
    #return t.zeros(size=(dim,), dtype=t.float32, device=dev)
    return t.rand(size=(dim,), dtype=t.float32, device=dev) * 10 - 5

def rand_fopt(dev = None):
    fopt = t.clip(t.distributions.cauchy.Cauchy(0, 100).sample(), -1000, 1000)
    if dev is not None:
        fopt = fopt.to(dev)
    return fopt

def seedable(func):
    def seed_wrapper(*args, **kargs):
        if 'seed' in kargs:
            t.manual_seed(kargs['seed'])
            del kargs['seed']
        return func(*args, **kargs)
    return seed_wrapper
