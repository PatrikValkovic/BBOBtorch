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

def random_optims(num, dim, localmax, globalmax, dev = None):
    optim = t.rand(size=(num, dim), dtype=t.float32, device=dev)
    optim[0] = optim[0] * 2 * globalmax - globalmax
    optim[1:] = optim[1:] * 2 * localmax - localmax
    return optim

def Cmatrix(num, setsize, dim, dev = None, first_pow = 1):
    set = t.pow(1000, 2 * t.arange(setsize, dtype=t.float32, device=dev) / setsize)
    diagonals = []
    l = Lambda(1000 ** first_pow, dim, t.float32, dev).T / pow(1000, 0.25)
    d = t.diag(l)
    p = t.randperm(dim)
    l = t.diag(d[p], out=l)
    diagonals.append(l.T)
    for i in range(num-1):
        alpha = float(set[t.randint(len(set), size=(1,))])
        l = Lambda(alpha, dim, t.float32, dev) / pow(alpha, 0.25)
        d = t.diag(l)
        p = t.randperm(dim)
        l = t.diag(d[p], out=l)
        diagonals.append(l.T)
    return t.stack(diagonals, dim=-1)

def wvector(num, dev):
    w = 1.1 + 8 * (t.arange(num, dtype=t.float32, device=dev) - 1) / num
    w[0] = 10
    return w


def seedable(func):
    def seed_wrapper(*args, **kargs):
        if 'seed' in kargs:
            t.manual_seed(kargs['seed'])
            del kargs['seed']
        return func(*args, **kargs)
    return seed_wrapper
