#!/usr/bin/env python

import math

import numpy as np
import scipy.stats
from scipy.stats._distn_infrastructure import _ShapeInfo
from scipy.special import kv
from scipy.optimize import minimize 
import warnings


def _isintegral(x):
    return x == np.round(x)


class sichel_gen(scipy.stats.rv_discrete):

    def _shape_info(self):
        return [_ShapeInfo("mu", False, (0, np.inf), (False, False)),
                _ShapeInfo("sigma", False, (0, np.inf), (False, False)),
                _ShapeInfo("v", False, (-np.inf, np.inf), (False, False))]

    def _rvs(self, mu, sigma, v, size=None, random_state=None):
        probs = random_state.random(size=size)
        max_prob = np.amax(probs)
        alpha = (1 / (sigma ** 2) + 2 * mu / sigma) ** 0.5
        w = (mu ** 2 + alpha ** 2)**0.5 - mu
        pmf = []
        cmf = []
        pos = len(pmf)
        while len(cmf) == 0 or cmf[-1] < max_prob:
            if pos > 1:
                p1 = pmf[pos - 2]
                p2 = pmf[pos - 1]
                pmf.append((2 * mu * w / alpha**2) * ((pos + v - 1) / pos) * p2 +
                           (mu * w / alpha) ** 2 / (pos * (pos - 1)) * p1)
                cmf.append(pmf[-1] + cmf[-1])
            elif pos == 0:
                pmf.append((w / alpha) ** v * kv(v, alpha) / kv(v, w))
                cmf.append(pmf[0])
            else:
                pmf.append(pmf[0] * (mu * w / alpha) * kv(v+1, alpha) / kv(v, alpha))
                cmf.append(pmf[1] + cmf[0])
            pos += 1
        cmfs = np.array(cmf, np.float64)
        return np.searchsorted(cmfs[1:], probs, side='left').astype(np.int64)

    def _argcheck(self, mu, sigma, v):
        return (mu > 0) and (sigma > 0)

    def _logpmf(self, x, mu, sigma, v):
        pmf = self._pmf(x, mu, sigma, v)
        logpmf = np.full(pmf.shape, -np.inf, pmf.dtype)
        where = np.where(pmf > 0)
        logpmf[where] = np.log(pmf[where])
        return logpmf

    def _pmf(self, x, mu, sigma, v):
        k = np.floor(x).astype(np.int64)
        maxobs = int(np.amax(k)) + 1
        if not issubclass(type(mu), float):
            mu = mu[0]
        if not issubclass(type(sigma), float):
            sigma = sigma[0]
        if not issubclass(type(v), float):
            v = v[0]

        alpha = (1 / (sigma ** 2) + 2 * mu / sigma) ** 0.5
        w = (mu ** 2 + alpha ** 2)**0.5 - mu
        pmf = np.empty(maxobs + 1, np.float64)
        for i in range(maxobs + 1):
            if i > 1:
                p1 = pmf[i - 2]
                p2 = pmf[i - 1]
                pmf[i] = ((2 * mu * w / alpha**2) * ((i + v - 1) / i) * p2 +
                          (mu * w / alpha) ** 2 / (i * (i - 1)) * p1)
            elif i == 0:
                pmf[i] = ((w / alpha) ** v * kv(v, alpha) / kv(v, w))
            else:
                pmf[i] = (pmf[0] * (mu * w / alpha) * kv(v+1, alpha) / kv(v, alpha))
        return pmf[k]

    def estimate_params(self, x, bounds, w=None):
        if w is None:
            mean = np.mean(x)
            var = np.var(x)
        else:
            mean = np.sum(x * w) / np.sum(w)
            var = np.sum(w * (x - mean) ** 2) / np.sum(w)
        w = mean / (var / mean -1)
        alpha2 = np.max([0.0001, (w + mean)**2 - mean**2])
        s = np.max([0.1, (math.sqrt(alpha2 + mean**2) + mean) / alpha2])
        params = {
            'mu': mean,
            'sigma': s,
            'v': -0.5,
        }
        return params

    def get_bounds(self, x):
        bounds = {
            'mu': (0.1, 5),
            'sigma': (0.1, 5),
            'v': (0.1, 5),
        }
        return bounds


class zasichel_gen(scipy.stats.rv_discrete):

    def _shape_info(self):
        return [_ShapeInfo("mu", False, (0, np.inf), (False, False)),
                _ShapeInfo("sigma", False, (0, np.inf), (False, False)),
                _ShapeInfo("v", False, (-np.inf, np.inf), (False, False)),
                _ShapeInfo("pi", False, (0, 1), (False, False))]

    def _rvs(self, mu, sigma, v, pi, size=None, random_state=None):
        probs = random_state.random(size=size)
        nz = probs > pi
        probs[nz] = (probs[nz] - pi) / (1 - pi)
        max_prob = np.amax(probs[nz])
        alpha = (1 / (sigma ** 2) + 2 * mu / sigma) ** 0.5
        w = (mu ** 2 + alpha ** 2)**0.5 - mu
        pmf = []
        cmf = []
        pos = len(pmf)
        while len(cmf) == 0 or cmf[-1] < max_prob:
            if pos > 1:
                p1 = pmf[pos - 2]
                p2 = pmf[pos - 1]
                pmf.append((2 * mu * w / alpha**2) * ((pos + v - 1) / pos) * p2 +
                           (mu * w / alpha) ** 2 / (pos * (pos - 1)) * p1)
                cmf.append(pmf[-1] + cmf[-1])
            elif pos == 0:
                pmf.append((w / alpha) ** v * kv(v, alpha) / kv(v, w))
                cmf.append(pmf[0])
            else:
                pmf.append(pmf[0] * (mu * w / alpha) * kv(v+1, alpha) / kv(v, alpha))
                cmf.append(pmf[1] + cmf[0])
            pos += 1
        cmfs = np.array(cmf, np.float64)
        values = np.zeros(size, np.float64)
        values[nz] = np.searchsorted(cmfs[1:], probs[nz], side='left').astype(np.int64)
        return values

    def _argcheck(self, mu, sigma, v, pi):
        return (mu > 0) and (sigma > 0) and (pi > 0) and (pi < 1)

    def _logpmf(self, x, mu, sigma, v, pi):
        pmf = self._pmf(x, mu, sigma, v, pi)
        logpmf = np.full(pmf.shape, -np.inf, pmf.dtype)
        where = np.where(pmf > 0)
        logpmf[where] = np.log(pmf[where])
        return logpmf

    def _pmf(self, x, mu, sigma, v, pi):
        k = np.floor(x).astype(np.int64)
        maxobs = int(np.amax(k)) + 1
        if not issubclass(type(mu), float):
            mu = mu[0]
        if not issubclass(type(sigma), float):
            sigma = sigma[0]
        if not issubclass(type(v), float):
            v = v[0]
        if not issubclass(type(pi), float):
            pi = pi[0]

        alpha = (1 / (sigma ** 2) + 2 * mu / sigma) ** 0.5
        w = (mu ** 2 + alpha ** 2)**0.5 - mu
        pmf = np.empty(maxobs + 1, np.float64)
        for i in range(maxobs + 1):
            if i > 1:
                p1 = pmf[i - 2]
                p2 = pmf[i - 1]
                pmf[i] = ((2 * mu * w / alpha**2) * ((i + v - 1) / i) * p2 +
                          (mu * w / alpha) ** 2 / (i * (i - 1)) * p1)
            elif i == 0:
                pmf[i] = ((w / alpha) ** v * kv(v, alpha) / kv(v, w))
            else:
                pmf[i] = (pmf[0] * (mu * w / alpha) * kv(v+1, alpha) / kv(v, alpha))
        pmf[0] = pi
        pmf[1:] *= (1 - pi)
        return pmf[k]

    def estimate_params(self, x, bounds, w=None):
        if w is None:
            mean = np.mean(x)
            var = np.var(x)
        else:
            mean = np.sum(x * w) / np.sum(w)
            var = np.sum(w * (x - mean) ** 2) / np.sum(w)
        w = mean / (var / mean -1)
        alpha2 = np.max([0.0001, (w + mean)**2 - mean**2])
        s = np.max([0.1, (math.sqrt(alpha2 + mean**2) + mean) / alpha2])
        params = {
            'mu': mean,
            'sigma': s,
            'v': -0.5,
            'pi': np.sum(x == 0) / x.size,
        }
        return params

    def get_bounds(self, x):
        bounds = {
            'mu': (0.1, 5),
            'sigma': (0.1, 5),
            'v': (0.1, 5),
            'pi': (0.001, 0.999),
        }
        return bounds


class zapoisson_gen(scipy.stats.rv_discrete):

    def _shape_info(self):
        return [_ShapeInfo("mu", False, (0, np.inf), (True, False)),
                _ShapeInfo("pi", False, (0, 1), (False, False))]

    def _rvs(self, mu, pi, size=None, random_state=None):
        zero_pmf = scipy.stats.poisson.pmf(k=0, mu=mu)
        probs = random_state.random(size=size)
        nz = np.where(probs >= pi - zero_pmf)
        values = np.zeros(size, dtype=np.int64)
        values[nz] = random_state.poisson(lam=mu, size=nz[0].shape[0])
        return values

    def _argcheck(self, mu, pi):
        return (mu >= 0) and (pi > 0) and (pi < 1)

    def _logpmf(self, k, mu, pi):
        nz = k > 0
        if not issubclass(type(mu), float):
            mu = mu[0]
        if not issubclass(type(pi), float):
            pi = pi[0]
        Pk = np.full(k.shape[0], np.log(pi), np.float64)
        Pk[nz] = scipy.stats.poisson.logpmf(k[nz], mu) + np.log(1-pi)
        return Pk

    def _pmf(self, k, mu, pi):
        return np.exp(self._logpmf(k, mu, pi))

    def estimate_params(self, x, bounds, w=None):
        if w is None:
            mean = np.mean(x)
        else:
            mean = np.sum(x * w) / np.sum(w)
        params = {
            "mu": mean,
            "pi": np.sum(x == 0) / x.size,
        }
        return params

    def get_bounds(self, x):
        bounds = {
            'mu': (0.1, np.amax(x) * 2),
            'pi': (0.001, 0.999),
        }
        return bounds


class zanbinom_gen(scipy.stats.rv_discrete):

    def _shape_info(self):
        return [_ShapeInfo("n", True, (1, np.inf), (True, False)),
                _ShapeInfo("p", False, (0, 1), (False, False)),
                _ShapeInfo("pi", False, (0, 1), (False, False))]

    def _rvs(self, n, p, pi, size=None, random_state=None):
        zero_pmf = scipy.stats.nbinom.pmf(k=0, p=p, n=n)
        probs = random_state.random(size=size)
        nz = np.where(probs >= pi - zero_pmf)
        values = np.zeros(size, dtype=np.int64)
        values[nz] = random_state.negative_binomial(p=p, n=n, size=nz[0].shape[0])
        return values

    def _argcheck(self, n, p, pi):
        return (n > 0) and _isintegral(n) and (pi > 0) and (pi < 1) and (p > 0) and (p < 1)

    def _logpmf(self, k, n, p, pi):
        nz = np.where(k > 0)
        if not issubclass(type(p), float):
            p = p[0]
        if issubclass(type(n), np.ndarray):
            n = n[0]
        if not issubclass(type(pi), float):
            pi = pi[0]
        Pk = np.full(k.shape, np.log(pi), np.float64)
        tmp = scipy.stats.nbinom.logpmf(k[nz], n, p) + np.log(1-pi)
        Pk[nz] = tmp
        return Pk

    def _pmf(self, k, n, p, pi):
        return np.exp(self._logpmf(k, n, p, pi))

    def estimate_params(self, x, bounds, w=None):
        if w is None:
            mean = np.mean(x)
            var = np.var(x)
        else:
            mean = np.sum(x * w) / np.sum(w)
            var = np.sum(w * (x - mean) ** 2) / np.sum(w)
        params = {
            'p': mean / var,
            'n': bounds['n'][0],
            # 'n': mean ** 2 / (var - mean),
            'pi': np.sum(x == 0) / x.size,
        }
        return params

    def get_bounds(self, x):
        maxval = np.amax(x)
        bounds = {
            'p': (0.001, 0.999),
            'n': (int(maxval), int(maxval)+1),
            'pi': (0.001, 0.999),
        }
        return bounds


class zabetanbinom_gen(scipy.stats.rv_discrete):

    def _shape_info(self):
        return [_ShapeInfo("n", True, (0, np.inf), (True, False)),
                _ShapeInfo("a", False, (0, np.inf), (False, False)),
                _ShapeInfo("b", False, (0, np.inf), (False, False)),
                _ShapeInfo("pi", False, (0, 1), (False, False))]

    def _rvs(self, n, a, b, pi, size=None, random_state=None):
        zero_pmf = scipy.stats.betanbinom.pmf(k=0, a=a, b=b, n=n)
        probs = random_state.random(size=size)
        nz = np.where(probs >= pi - zero_pmf)
        values = np.zeros(size, dtype=np.int64)
        Ps = random_state.beta(a=a, b=b, size=nz[0].shape[0])
        values[nz] = random_state.negative_binomial(p=Ps, n=n, size=nz[0].shape[0])
        return values

    def _argcheck(self, n, a, b, pi):
        return (n >= 0) and _isintegral(n) and (pi > 0) and (pi < 1) and (a > 0) and (b > 0)

    def _logpmf(self, k, n, a, b, pi):
        nz = np.where(k > 0)
        if not issubclass(type(a), float):
            a = a[0]
        if not issubclass(type(b), float):
            b = b[0]
        if issubclass(type(n), np.ndarray):
            n = n[0]
        if not issubclass(type(pi), float):
            pi = pi[0]
        Pk = np.full(k.shape, np.log(pi), np.float64)
        tmp = scipy.stats.betanbinom.logpmf(k[nz], n, a, b) + np.log(1-pi)
        Pk[nz] = tmp
        return Pk

    def _pmf(self, k, n, a, b, pi):
        return np.exp(self._logpmf(k, n, a, b, pi))

    def estimate_params(self, x, bounds, w=None):
        if w is None:
            mean = np.mean(x)
            var = np.var(x)
        else:
            mean = np.sum(x * w) / np.sum(w)
            var = np.sum(w * (x - mean) ** 2) / np.sum(w)
        if mean > var:
            var = mean * 1.25
        p = min(1, mean / var)
        v = (p * (1 - p) / 0.03) - 1
        params = {
            # 'n': int(max(1.0, mean**2 / (var-mean))),
            'n': bounds['n'][0],
            'a': max(1.01, mean * v),
            'b': max(0.01, (1 - mean) * v),
            'pi': np.sum(x == 0) / x.size,
        }
        return params

    def get_bounds(self, x):
        maxval = np.amax(x)
        bounds = {
            'a': (2.01, 10),
            'b': (0.01, 10),
            'n': (int(maxval), int(maxval)+1),
            'pi': (0.001, 0.999),
        }
        return bounds


class binom_gen(scipy.stats._discrete_distns.binom_gen):

    def estimate_params(self, x, bounds, w=None):
        if w is None:
            mean = np.mean(x)
        else:
            mean = np.sum(x * w) / np.sum(w)
        maxval = np.amax(x)
        params = {
            'p': mean / maxval,
            'n': bounds['n'][0], 
        }
        return params

    def get_bounds(self, x):
        maxval = np.amax(x)
        bounds = {
            'p': (0.001, 0.999),
            'n': (int(maxval), int(maxval)+1),
        }
        return bounds


class nbinom_gen(scipy.stats._discrete_distns.nbinom_gen):

    def estimate_params(self, x, bounds, w=None):
        if w is None:
            mean = np.mean(x)
            var = np.var(x)
        else:
            mean = np.sum(x * w) / np.sum(w)
            var = np.sum(w * (x - mean) ** 2) / np.sum(w)
        params = {
            # 'n': mean ** 2 / (var - mean),
            'n': bounds['n'][0],
            'p': mean / var,
        }
        return params

    def get_bounds(self, x):
        maxval = np.amax(x)
        bounds = {
            'p': (0.001, 0.999),
            'n': (int(maxval), int(maxval)+1),
        }
        return bounds


class poisson_gen(scipy.stats._discrete_distns.poisson_gen):

    def estimate_params(self, x, bounds, w=None):
        if w is None:
            mean = np.mean(x)
        else:
            mean = np.sum(x * w) / np.sum(w)
        params = {
            'mu': mean,
        }
        return params

    def get_bounds(self, x):
        maxval = np.amax(x)
        bounds = {
            'mu': (0.1, maxval * 2),
        }
        return bounds


class betabinom_gen(scipy.stats._discrete_distns.betabinom_gen):

    def estimate_params(self, x, bounds, w=None):
        maxval = np.amax(x)
        if w is None:
            mean = np.mean(x)
            var = np.var(x)
        else:
            mean = np.sum(x * w) / np.sum(w)
            var = np.sum(w * (x - mean) ** 2) / np.sum(w)
        pi = mean / maxval
        theta = (var - mean * (1 - pi)) / (maxval * mean * (1 - pi) - var)
        params = {
            'a': pi / theta,
            'b': (1 - pi) / theta,
            'n': bounds['n'][0],
        }
        return params

    def get_bounds(self, x):
        maxval = np.amax(x)
        bounds = {
            'a': (2.01, 20),
            'b': (2.01, 20),
            'n': (int(maxval), int(maxval)+1),
        }
        return bounds


class betanbinom_gen(scipy.stats._discrete_distns.betanbinom_gen):

    def estimate_params(self, x, bounds, w=None):
        maxval = np.amax(x)
        if w is None:
            mean = np.mean(x)
            var = np.var(x)
        else:
            mean = np.sum(x * w) / np.sum(w)
            var = np.sum(w * (x - mean) ** 2) / np.sum(w)
        if mean > var:
            var = mean * 1.25
        p = mean / var
        var2 = 0.05 ** 2
        v = p * (1 - p) / var2 - 1
        params = {
            # 'n': mean * mean / (var - mean),
            'n': bounds['n'][0],
            'a': p * v,
            'b': (1 - p) * v,
        }
        if params['b'] > params['n']:
            params['b'], params['n'] = params['n'], params['b']
        return params

    def get_bounds(self, x):
        maxval = np.amax(x)
        bounds = {
            'a': (2.01, 10),
            'b': (0.01, 100),
            'n': (int(maxval), int(maxval)+1),
        }
        return bounds

class Distribution():
    valid_dists = {
        "BI"  : "binomial",
        "PO"  : "poisson",
        "NB"  : "negative binomial",
        "BB"  : "beta-binomial",
        "BNB" : "beta-negative binomial",
        "SI"  : "sichel",
        "ZP"  : "zero-adjusted poisson",
        "ZNB" : "zero-adjusted negative binomial",
        "ZBNB": "zero-adjusted beta negative binomial"
    }

    def __init__(self, name, random_state=None):
        if name not in Distribution_dict:
            raise KeyError(f"The distribution {name} isn't a distribution option")
        if random_state is None:
            random_state = numpy.random.default_rng()
        self.RNG = random_state
        self.dist = Distribution_dict[name]
        self.short_name = name
        self.name = self.valid_dists[name]
        self.bounds = {}
        self.params = {}
        param_shapes = self.dist._shape_info()
        for shape in param_shapes:
            name = shape.name
            integrality = shape.integrality
            domain = shape.domain
            self.bounds[name] = domain
            if integrality:
                if domain[0] == -np.inf:
                    start = np.iinfo(int).min
                else:
                    start = int(np.ceil(domain[0]))
                if domain[1] == np.inf:
                    end = np.iinfo(int).max
                else:
                    end = int(np.floor(domain[1])) + 1
                self.params[name] = random_state.integers(start, end)
            else:
                self.params[name] = domain[0] + random_state.random() * (domain[1] - domain[0])
        return

    def set_parameters(self, **kwargs):
        self.check_parameters(**kwargs)
        for k, v in kwargs.items():
            self.params[k] = v
        return

    def check_parameters(self, **kwargs):
        if not self.dist._argcheck(**kwargs):
            raise ValueError(f"The parameters {kwargs} don't match the distribution {self.name}")
        param_shapes = self.dist._shape_info()
        for ps in param_shapes:
            if ps.integrality:
                kwargs[ps.name] = int(kwargs[ps.name])
        return kwargs

    def estimate_params(self, X, weights=None):
        self.params = self.dist.estimate_params(X, self.bounds, weights)
        for p in self.params.keys():
            self.params[p] = max(self.bounds[p][0], min(self.bounds[p][1], self.params[p]))
        return self.params

    def optimize_parameters(self, X, W=None):
        warnings.filterwarnings("ignore")
        if self.bounds is None:
            self.bounds = self.dist.get_bounds(X)
        est_params = self.estimate_params(X, W)
        if W is None:
            fit = scipy.stats.fit(self.dist, X, self.bounds, guess=est_params)
            new_params = {k: v for k, v in fit.params._asdict().items() if k in est_params}
        else:
            def LLH(params, param_names, X, W):
                param_dict = {param_names[x]: params[x] for x in range(len(params))
                              if param_names[x] != 'n'}
                if 'n' in param_names:
                    index = param_names.index('n')
                    return -np.sum(self.dist.logpmf(X, n=params[index], **param_dict) * W)
                else:
                    return -np.sum(self.dist.logpmf(X, **param_dict) * W)

            params = []
            param_names = []
            bounds = []
            for n, v in est_params.items():
                params.append(v)
                param_names.append(n)
                bounds.append(self.bounds[n])
            fit = scipy.optimize.minimize(LLH, np.array(params),
                                          args=(param_names, X, W), bounds=bounds)
            new_params = {param_names[x]: fit.x[x] for x in range(len(params))}
            if 'n' in self.params:
                new_params['n'] = self.params['n']
        new_params = self.check_parameters(**new_params)
        self.params = new_params
        return new_params

    def get_bounds(self, X):
        self.bounds = self.dist.get_bounds(X)

    def rvs(self, size=None, **kwargs):
        if len(kwargs) == 0:
            return self.dist.rvs(size=size, random_state=self.RNG, **self.params)
        else:
            return self.dist.rvs(size=size, random_state=self.RNG, **kwargs)

    def pmf(self, X, **kwargs):
        if len(kwargs) == 0:
            return self.dist.pmf(X, **self.params)
        else:
            return self.dist.pmf(X, **kwargs)

    def logpmf(self, X, **kwargs):
        if len(kwargs) == 0:
            return self.dist.logpmf(X, **self.params)
        else:
            return self.dist.logpmf(X, **kwargs)


Distribution_dict = {
    "BI"  : binom_gen(name="binom"),
    "PO"  : poisson_gen(name='poisson'),
    "NB"  : nbinom_gen(name='nbinom'),
    "BB"  : betabinom_gen(name='betabinom'),
    "BNB" : betanbinom_gen(name='betanbinom'),
    "SI"  : sichel_gen(name="sichel"),
    "ZP"  : zapoisson_gen(name="zapoisson"),
    "ZNB" : zanbinom_gen(name="zanbinom"),
    "ZBNB": zabetanbinom_gen(name="zabetanbinom"),
    "ZS"  : zasichel_gen(name="zasichel"),
}
