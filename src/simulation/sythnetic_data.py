#!/usr/bin/env python
# coding: utf-8
"""
@Author  :   C.Z. Tang
"""

import os
from abc import ABCMeta
from cmath import pi
from typing import Tuple, TypeVar, Iterable
import numpy as np
import random
import math
import pandas as pd
from sklearn.datasets import make_blobs, make_regression
from pyhocon import HOCONConverter, ConfigFactory
import scipy.stats as stats


Matrix = TypeVar('Matrix', pd.DataFrame, np.ndarray)


class Simulator(metaclass=ABCMeta):
    def __init__(self, conf, **kwargs):
        self.conf = conf


class FeatGenerator(Simulator):
    def __init__(self, conf, **kwargs):
        self.conf = conf
        self.n_obs = kwargs.pop('n_obs', conf.get_int('n_obs'))
        self.n_unobs = kwargs.pop('n_unobs', conf.get_int('n_unobs'))
        self.n_period = kwargs.pop('n_period', conf.get_int('n_period'))
        self.n_treat = kwargs.pop('n_treat', conf.get_int('n_treat'))
        self.confouders = kwargs.pop('confouders', conf.get_list('confouders'))
        self.data_path = kwargs.pop('data_path', conf.get_list('data.path'))
        self.rand_assignment = kwargs.pop('rand_assignment', conf.get_bool('rand_assignment'))
        self.bias_ratio = kwargs.pop('bias_ratio', conf.get_float('bias_ratio', 0))
        self.treat_rates = []
        self.treat_values = []
        treatment_info = kwargs.pop('treatment_info', conf.get_list('treatment_info'))
        for it in treatment_info:
            self.treat_values.append(it[0])
            self.treat_rates.append(it[1])
        self.treat_rates = np.array(self.treat_rates)
        self.treat_values = np.array(self.treat_values)
        self.rand_state = 9

        # fix random seed
        np.random.seed(self.rand_state)
        random.seed(self.rand_state)

    def save(self, X: Matrix, w: Matrix, y: Matrix, y0: Matrix, eff: Matrix, path: list = None) -> None:
        """[summary]

        Args:
            X (Matrix): n * d
            y (Matrix): n * 2*p
            path (str): [description]
        """
        cols = [f'conf_{i}' for i in range(self.n_obs)] + [f'cov_{i}' for i in range(self.n_obs)] + ['treatment']
        features = pd.DataFrame(np.concatenate([X, np.expand_dims(w, 1)], axis=-1), columns=cols)
        mcols = pd.MultiIndex.from_arrays([['y'] * self.n_period +
                                           ['y0'] * self.n_period + ['eff'] * len(self.treat_values),
                                           [i for i in range(self.n_period)] * 2 + list(self.treat_values)])
        targets = pd.DataFrame(np.concatenate([y, y0, eff], axis=1), columns=mcols)
        if path is None:
            return targets, features
        dir = os.path.dirname(path[0])
        if os.path.exists(dir) is False:
            os.mkdir(dir)
            print(f'{dir} has been created.')
        # train test 划分
        tr_ratio = self.conf.get_float('data.train_ratio', 1)
        n_ins, _ = features.shape
        tr_ind = (np.random.RandomState(self.rand_state).permutation(n_ins) < int(n_ins * tr_ratio))
        base_dir = os.path.dirname(path[0])
        feature_name = os.path.basename(path[0])
        target_name = os.path.basename(path[1])
        if tr_ratio < 1:
            te_ind = ~tr_ind
            features.loc[te_ind].to_csv(os.path.join(base_dir, 'test_' + feature_name))
            targets.loc[te_ind].to_csv(os.path.join(base_dir, 'test_' + target_name))
        features.loc[tr_ind].to_csv(path[0])
        targets.loc[tr_ind].to_csv(path[1])
        conf = self.conf.copy()
        conf.put('feature', cols[:-1])
        conf.put('treatment', cols[-1])
        conf.put('periods', [i for i in range(self.n_period)])
        conf.put('treat_dt', self.n_period - self.n_treat)
        conf.put('target', ['y'])
        f = open(f'{os.path.join(dir, "synthetic.conf")}', 'w')
        f.write(HOCONConverter.convert(conf))
        return

    def generate(self, **kwargs):
        # step 1: generate cross-section features
        n_cluster = kwargs.pop('n_clusters')
        n_sample = kwargs.pop('n_samples')
        centers = np.random.binomial(2, 0.6, [n_cluster, self.n_unobs])
        stds =  np.random.normal(2, .5, n_cluster)
        conf = self.conf
        intercept_ms = conf.get_list('intercept_ms', [0, 1])
        y00_ms = conf.get('y00_ms', [0, 1])
        # 随机分配 clusters for treatment
        cluster_samples = np.zeros(n_cluster, int)
        assert n_cluster % len(self.treat_values) == 0, '#clusters must be divisible by #treatment'
        for v, r in zip(self.treat_values, self.treat_rates):
            steps = int(n_cluster//len(self.treat_values))
            for i in range(steps):
                cluster_samples[v*steps+i] += int(r*n_sample)/steps
        cluster_samples[-1] = n_sample - np.sum(cluster_samples[:-1])

        X_uno, uid = make_blobs(n_samples=cluster_samples,
                              n_features=self.n_unobs,
                              centers=centers,
                              random_state=self.rand_state,
                              cluster_std=stds,
                              center_box=(-5, 5))
        centers = np.random.binomial(2, 0.6, [n_cluster, self.n_obs])
        X_obs, oid = make_blobs(n_samples=cluster_samples,
                              n_features=self.n_obs,
                              centers=centers,
                              random_state=self.rand_state,
                              cluster_std=stds,
                              center_box=(-5, 5))
        X = np.concatenate([X_uno, X_obs], axis=-1)

        # step 2: generate treatment and assign
        if self.rand_assignment or self.bias_ratio == 0:
            w_ind = self.assign_treatments(np.random.random(n_sample))
        else:
            # w_ind = self.assign_treatments(X_uno, uid)
            step = n_cluster//len(self.treat_values)
            cluster_map = np.array(uid, dtype=int)
            if n_cluster != len(self.treat_values):
                 cluster_map = cluster_map // step
            w_ind = self.assign_treatments_by_clusters(cluster_map, self.bias_ratio)
        w = self.treat_values[w_ind]
        w_derive = self.treatment_derive_features(np.tile(self.treat_values, [n_sample, 1]).T.reshape(-1), mean=1)
        _z = np.concatenate([np.tile(X, [len(self.treat_values), 1]), w_derive], axis=-1)
        _w = np.tile(self.treat_values, [n_sample, 1]).T.reshape(-1)
        eff_rate = _w / _w.max()
        use_onlinear_eff = True
        if use_onlinear_eff is False:
            # linear treatment effect
            _dy = np.squeeze(self.linear_regression(_z, relative=1.5/_z.shape[1]))
        else:
            # nolinear treatment effect
            part1 = self.make_friedman1(_z, None)
            part2 = self.make_friedman2(_z, None)
            part1 = part1/10
            part2 = np.log(abs(part2)+1)
            _dy = np.squeeze(part1 * part2)
        _dy = self.add_noise(eff_rate * _dy, 0, 1).reshape([len(self.treat_values), -1]).T
        _dy[:,0] = 0
        # _dy[:,:] = 0
        dy = _dy[np.arange(w_ind.shape[0]), w_ind]

        # step 2: generate the intercept
        intercept = self.make_friedman1(X, oid) + self.make_friedman2(X, oid)
        intercept = (intercept - intercept.mean()) / intercept.std() * intercept_ms[1] + intercept_ms[0]
        # step 3: generate the y_{t=0}^0
        y00 = self.linear_regression(X)
        y00 = (y00 - y00.mean()) + y00_ms[0]
        lambd = np.stack([self.time_series(avg_amplitude=5, avg_pattern_length=7) for _ in range(3)], axis=1)
        temp_w = self.linear_regression(X_obs, outd=3)
        y_ts = np.matmul(temp_w, lambd.T)
        # step 3.3: y_0 = y_ts+intercept+y_{t=0}^0
        y0 = y00[:, 0]
        ys = []
        p_mean = 0.8
        # p = stats.truncnorm.rvs((0-p_mean)/0.3, (1-p_mean)/0.3, p_mean, 0.3, n_sample)
        p = p_mean
        for _ts in y_ts.T:
            y0 = (_ts + intercept) + p * y0
            ys.append(y0)
        # y0 = y_ts + np.expand_dims(intercept, axis=1) + y00
        y0 = np.column_stack(ys[-self.n_period:])
        y = np.copy(y0)
        y[:, -self.n_treat:] += dy[:, np.newaxis]
        # # shuffle data
        # sf_ind = np.arange(n_sample)
        # np.random.shuffle(sf_ind)
        self.save(X, w, y, y0, _dy, self.data_path)

    def time_series(self,
                    data=None,
                    length=100,
                    avg_pattern_length=7,
                    variance_pattern_length=3,
                    avg_amplitude=1,
                    variance_amplitude=2,
                    default_variance=1,
                    include_negatives=True):
        def generate_bell(length, amplitude, default_variance):
            return np.random.normal(0, default_variance, length) + amplitude * np.arange(length) / length

        def generate_funnel(length, amplitude, default_variance):
            return np.random.normal(0, default_variance, length) + amplitude * np.arange(length)[::-1] / length

        def generate_cylinder(length, amplitude, default_variance):
            return np.random.normal(0, default_variance, length) + amplitude

        length = max(self.n_period*3, length)
        generators = [generate_bell, generate_funnel, generate_cylinder]
        data = np.random.normal(0, default_variance, length)
        length = len(data)
        current_start = random.randint(0, avg_pattern_length)
        current_length = current_length = max(1, math.ceil(random.gauss(avg_pattern_length, variance_pattern_length)))

        while current_start + current_length < length:
            generator = random.choice(generators)
            current_amplitude = random.gauss(avg_amplitude, variance_amplitude)

            while current_length <= 0:
                current_length = -(current_length - 1)
            pattern = generator(current_length, current_amplitude, default_variance)

            if include_negatives and random.random() > 0.5:
                pattern = -1 * pattern

            data[current_start:current_start + current_length] += pattern

            current_start = current_start + current_length + random.randint(0, avg_pattern_length)
            current_length = max(1, math.ceil(random.gauss(avg_pattern_length, variance_pattern_length)))
        return np.array(data)

    def make_friedman1(self, X: Matrix, y: Matrix) -> Matrix:
        """y(X) = 10 * sin(pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]

        Args:
            X (Matrix): [description]
            y (Matrix): [description]
            mean (float, optional): [description]. Defaults to None.
            std (float, optional): [description]. Defaults to None.

        Returns:
            Matrix: [description]
        """
        pi = np.math.pi
        _, d = X.shape
        assert d > 4, 'you must support more than 4-dimentional feature!!!'
        out = 0
        count = 0
        for i in range(0, d - 4):
            out += 10 * np.sin(
                pi * X[:, i] * X[:, i + 1]) + 20 * (X[:, i + 2] - 0.5)**2 + 10 * X[:, i + 3] + 5 * X[:, i + 4]
            count += 1
        out = out / count
        if y is not None:
            n_label = y.max()
            coeff = np.exp(y / n_label)
            out = np.multiply(out, coeff)
        return out

    def make_friedman2(self, X: Matrix, y: Matrix) -> Matrix:
        """y(X) = (X[:, 0] ** 2 + (X[:, 1] * X[:, 2]  - 1 / (X[:, 1] * X[:, 3])) ** 2) ** 0.5 + noise * N(0, 1)
        Args:
            X (Matrix): [description]
            y (Matrix): [description]
            mean (float, optional): [description]. Defaults to None.
            std (float, optional): [description]. Defaults to None.

        Returns:
            Matrix: [description]
        """
        _, d = X.shape
        assert d > 4, 'you must support more than 4-dimentional feature!!!'
        out, count = 0, 0
        for i in range(0, d - 4):
            out += (X[:, i]**2 + (X[:, i + 1] * X[:, i + 2] - 1 / (X[:, i + 1] * X[:, i + 3]))**2)**0.5
            count += 1
        out = out / count
        if y is not None:
            n_label = y.max()
            coeff = np.exp(y / n_label)
            out = np.multiply(out, coeff)
        return out

    def init_y0(self, X: Matrix, y: Matrix, mean: float = None, std: float = None) -> Matrix:
        out = self.make_friedman2(X, y)
        if mean is not None and std is not None:
            out = self.add_noise(out, 0, 0.3 * std)
        if mean is not None and std is not None:
            # 标准化
            out = (out - out.mean()) / out.std()
            out = out * std + mean
        return out

    def linear_regression(self, X: Matrix, y: Matrix = None, relative: Tuple[int, dict] = 0, outd=1) -> Matrix:
        assert y is None or isinstance(relative, Iterable)
        n, d = X.shape
        if y is None:
            w = np.random.normal(loc=relative, scale=1, size=[d, outd])
            return np.matmul(X, w)
        uniq_ys = np.unique(y)
        out = np.zeros([n, outd])
        for i in uniq_ys:
            ind = (y==i)
            out[ind] = self.linear_regression(X[ind], relative=relative[i], outd=outd)
        return out

    def gaisson_regression(self, X: Matrix, y: Matrix= None, relative: Tuple[int, list] = 0, outd=1)-> Matrix:
        out = self.linear_regression(X, y, relative, outd)
        return np.exp(-out**2)

    def logistic_regression(self, X: Matrix, y: Matrix = None, relative: int = 0) -> Matrix:
        logit = self.linear_regression(X, y, relative)
        return 1 / (1 + np.exp(-logit))

    def truncat(self, X: Matrix, min: float, max: float) -> Matrix:
        return X.clip(min, max)

    def add_noise(self, X: Matrix, mean: Matrix, std: Matrix, y: Matrix = None):
        noise = np.random.normal(0, 1, X.shape)
        if y is not None:
            n_label = y.max()
            coeff = np.clip(np.exp(y / n_label), 1, 2)
            if len(coeff.shape) < len(noise.shape):
                coeff = np.expand_dims(coeff, axis=-1)
            noise = noise * coeff
        noise = (noise - noise.mean()) / noise.std() * std + mean
        return X + noise

    def assign_treatments(self, X: Matrix, bias: Matrix = 0):
        """assign treatment

        Args:
            X (Matrix): confounding covariates
            bias (Matrix, optional): . Defaults to None.

        Returns:
            [type]: [description]
        """
        treat_size = len(self.treat_rates)
        if isinstance(bias, Iterable) and len(bias.shape) >= 2:
            bias = bias[:, 0]
        if len(X.shape) > 1:
            weight = self.gaisson_regression(X)[:,0]
            score = self.make_friedman1(X, None)*weight+(1-weight)*self.make_friedman2(X, None) 
            score = (score - score.mean()) / score.std()
            score = (1 / (1 + np.exp(-score)))  # (score - score.min()) / (score.max() - score.min())
        else:
            score = np.expand_dims(X, -1)
        if treat_size == 0:  # 连续的
            pass
        else:
            cdf = np.cumsum(self.treat_rates) / np.sum(self.treat_rates)
            cut_points = np.percentile(score, 100 * cdf)
            cut_points[-1] = np.Inf
            score = self.add_noise(score, 0, score.std())
            ps = np.clip(score, 0, 1)
            if len(ps.shape) == 1:
                ps = ps[:,np.newaxis]
            w = np.apply_along_axis(lambda x: np.where(cut_points > x)[0][0], 1, ps)
        return w

    def assign_treatments_by_clusters(self, cluster_ids: Matrix, r:float=0.3):
        assert r > 0 and r <= 1, 'r must be positive, between (0, 1]'
        # step 1 保持r比例的不变，即在 i 聚类簇里面的样本中 r 比例的被施加 i treatment。
        n = cluster_ids.shape[0]
        rem_ind = (np.random.permutation(n) < int(r*n))
        w = np.zeros(n, dtype=np.int32)
        w[rem_ind] = cluster_ids[rem_ind]
        # step 2 剩余1-r 的样本随机按比例分配
        if r < 1:
            w[~rem_ind] = self.assign_treatments(np.random.random(n-int(r*n)))
        return w

    def treatment_derive_features(self, w: Matrix, mean=0, std=1) -> Matrix:
        feat = np.stack([w, w**2, np.log(np.abs(w) + 1), np.sin(pi * w), np.exp(0.1 * w)], axis=1)
        return (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-6) * std + mean


if __name__ == "__main__":
    conf = ConfigFactory.parse_file('config/synth_binary.conf')
    conf.put('$.bias_ratio', 0.7)
    fg = FeatGenerator(conf)
    fg.generate(n_samples=100000, n_clusters=2)
