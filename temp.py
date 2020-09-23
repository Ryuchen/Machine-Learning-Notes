# !/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# @Author: Ryuchen
# @Time: 2020/09/23-11:35
# @Site: https://ryuchen.github.io
# @Contact: chenhaom1993@hotmail.com
# @Copyright: Copyright (C) 2019-2020 Machine-Learning-Notes.
# ========================================================
"""
...
DocString Here
...
"""
import random
import numpy as np

from scipy import stats


class MultivariateGaussianMixture:
    """ Multivariate Gaussians Mixture Model with EM estimation """

    def __init__(self, data, rtol=1e-3, n_cluster=3, max_iter=100, restarts=30):
        """
        Starts with random initialization *restarts* times
        Runs optimization until saturation with *rtol* reached
        or *max_iter* iterations were made.
        X: (N, d), data points
        C: int, number of clusters
        """
        self.data = data
        self.N = data.shape[0]  # number of objects
        self.d = data.shape[1]  # dimension of each object
        self.C = n_cluster  # number of clusters
        self.rtol = rtol

        self.best_loss = None
        self.best_pi = None
        self.best_mu = None
        self.best_sigma = None

        self.pis = {}
        self.mus = {}
        self.sigmas = {}
        self.losses = {}

        self.max_iter = max_iter
        self.restarts = restarts

    def E_step(self, pi, mu, sigma):
        """
        Performs E-step on GMM model
        Each input is numpy array:
        X: (N x d), data points
        pi: (C), mixture component weights
        mu: (C x d), mixture component means
        sigma: (C x d x d), mixture component covariance matrices
        Returns:
        gamma: (N x C), probabilities of clusters for objects
        """
        gamma = np.zeros((self.N, self.C))
        for i in range(self.C):
            gamma[:, i] = pi[i] * stats.multivariate_normal(mean=mu[i], cov=sigma[i]).pdf(self.data)
        for n in range(self.N):
            z = np.sum(gamma[n, :])
            gamma[n, :] = gamma[n, :] / z
        return gamma

    def M_step(self, gamma):
        """
        Performs M-step on GMM model
        Each input is numpy array:
        X: (N x d), data points
        gamma: (N x C), distribution q(T)
        Returns:
        pi: (C)
        mu: (C x d)
        sigma: (C x d x d)
        """
        mu = []
        sigma = []
        pi = []
        reg_cov = 1e-6 * np.identity(len(self.data[0]))
        for i in range(self.C):
            r_x = np.sum(self.data * gamma[:, i].reshape(len(self.data), 1), axis=0)
            r_i = sum(gamma[:, i])
            mu.append(r_x / r_i)
            s = ((1 / r_i) * np.dot((np.array(gamma[:, i]).reshape(len(self.data), 1) * (self.data - mu[i])).T, (self.data - mu[i]))) + reg_cov
            sigma.append(s)
            pi.append(r_i / np.sum(gamma))
        return np.array(pi), np.array(mu), np.array(sigma)

    def compute_vlb(self, pi, mu, sigma, gamma):
        """
        Each input is numpy array:
        X: (N x d), data points
        gamma: (N x C), distribution q(T)
        pi: (C)
        mu: (C x d)
        sigma: (C x d x d)
        Returns value of variational lower bound
        """

        L1 = []
        L2 = []

        for i in range(self.data.shape[0]):
            for j in range(self.C):
                Norm = stats.multivariate_normal(mean=mu[j], cov=sigma[j]).pdf(self.data)
                if np.logical_and(not np.isnan(Norm[i]), not np.isinf(Norm[i])):
                    if Norm[i] > 0:
                        L1.append(gamma[i, j] * (np.log(pi[j]) + np.log(Norm[i])))
                    if gamma[i, j] > 0:
                        L2.append((gamma[i, j] * np.log(gamma[i, j])))
        L1 = np.array(L1)
        L2 = np.array(L2)
        L1 = L1[~np.isnan(L1)]
        L2 = L2[~np.isnan(L2)]
        loss = np.sum(L1) - np.sum(L2)

        return loss

    def create_mean(self):
        mean = np.random.randint(min(self.data[:, 0]), max(self.data[:, 0]), size=(self.C, len(self.data[0])))
        return mean

    def create_sigma(self):
        I = np.diag((1, 1, 1, 1))
        sigmas = [I, I, I]
        sigmas = np.array(sigmas)
        return sigmas

    def create_pi(self):
        a = []
        pi_1 = []
        for i in range(self.C):
            a.append(random.randint(1, 10))
        z = sum(a)
        for x in range(self.C):
            pi_1.append(a[x] / z)
        return np.array(pi_1)

    def best_config(self):
        for key, value in self.losses.items():
            max_val = max(list(self.losses.values()))
            if value == max_val:
                return self.losses[key], self.pis[key], self.mus[key], self.sigmas[key]

    def train(self):
        for _ in range(self.restarts):
            try:
                # creating parameters
                mu = self.create_mean()
                sigma = self.create_sigma()
                pi = self.create_pi()

                # The Functions
                stop_point = False
                loss = 1
                for ind in range(self.max_iter):
                    if not stop_point:
                        if np.logical_and(not np.isnan(mu).any(), not np.isinf(mu).any()):
                            if np.logical_and(not np.isnan(sigma).any(), not np.isinf(sigma).any()):
                                gamma = self.E_step(pi, mu, sigma)
                                pi, mu, sigma = self.M_step(gamma)
                                if ind > 0 and self.rtol >= np.abs((self.compute_vlb(pi, mu, sigma, gamma) - loss) / loss):
                                    stop_point = True
                                else:
                                    loss = self.compute_vlb(pi, mu, sigma, gamma)
                    else:
                        break
                self.losses[_] = loss
                self.pis[_] = pi
                self.mus[_] = mu
                self.sigmas[_] = sigma

            except Exception:
                print("array must not contain infs or NaNs")
                pass

        best_loss, best_pi, best_mu, best_sigma = self.best_config()

        return best_loss, best_pi, best_mu, best_sigma


if __name__ == '__main__':
    import seaborn as sns

    penguins = sns.load_dataset("penguins")
    penguins = penguins.dropna()

    MGMM = MultivariateGaussianMixture(penguins.iloc[:, 2:-1].values, 3)

    MGMM.train()
