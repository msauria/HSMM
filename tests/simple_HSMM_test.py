#!/usr/bin/env python

import sys
import multiprocessing

import numpy
import HSMM
import matplotlib.pyplot as plt
import scipy.stats
from scipy.special import logsumexp

def main():

    TM = numpy.array([[0.9, 0.1], [0.05, 0.95]], numpy.float64)
    param = [[{"n": 1, "p": 0.1}], [{"n": 1, "p": 0.9}]]
    with HSMM.HSMM(
            num_states=2,
            num_threads=1,
            distributions=["BI"],
            transition_matrix=TM,
            initial_probabilities=None,
            seed=2001) as model:
        model.set_dist_parameters(param)
        model.set_dwell_times([2, 6])
        model.save('test_model.npz')

    with HSMM.HSMM(
            fname='test_model.npz',
            seed=2001) as model:
        # print(model)
        seqs, states = model.generate_sequences(100, 1000)
        dwells = [[], []]
        for i, seq in enumerate(seqs):
            breaks = numpy.r_[0, numpy.where(numpy.diff(states[i] < 2))[0] + 1, seq.shape[0]]
            for j in range(breaks.shape[0] - 1):
                s, e = breaks[j:j+2]
                dwells[int(states[i][s] >= 2)].append(e - s)
        fig, ax = plt.subplots(1,1)
        Y = numpy.array(dwells[0]) - 1
        maxval = numpy.amax(Y)
        Y = numpy.bincount(Y)/Y.shape[0]
        ax.plot(numpy.arange(maxval + 1), Y, color='red')
        ax.plot(numpy.arange(maxval), scipy.stats.nbinom.pmf(numpy.arange(maxval), 2, 0.1), color='red', linestyle='dashed')

        Y = numpy.array(dwells[1]) - 1
        maxval = numpy.amax(Y)
        Y = numpy.bincount(Y)/Y.shape[0]
        ax.plot(numpy.arange(maxval + 1), Y, color='blue')
        ax.plot(numpy.arange(maxval), scipy.stats.nbinom.pmf(numpy.arange(maxval), 6, 0.05), color='blue', linestyle='dashed')
        plt.savefig('test_dwells.pdf')
        plt.close()

        model.load_observations(seqs)
        model.train(maxIterations=1)
        dindices = numpy.r_[0, numpy.cumsum(model.dwells)]
        pre_gamma = model.lngamma
        gamma = numpy.zeros((pre_gamma.shape[0], 2), numpy.float64)
        for i in range(dindices.shape[0] - 1):
            s, e = dindices[i:i+2]
            gamma[:, i] = numpy.exp(logsumexp(pre_gamma[:, s:e], axis=1))
        gamma /= numpy.sum(gamma, axis=1, keepdims=True)

        breaks = numpy.r_[0, numpy.where(numpy.diff(gamma[:, 0] > gamma[:, 1]))[0] + 1, gamma.shape[0]]
        counts = numpy.zeros((1000, 2), numpy.float64)
        for i in range(breaks.shape[0] - 1):
            s, e = breaks[i:i+2]
            if gamma[s, 0] > gamma[s, 1]:
                index = 0
            else:
                index = 1
            span = e - s - 1
            if span >= counts.shape[0]:
                continue
            counts[span, index] += numpy.mean(gamma[s:e, index])
        counts /= numpy.sum(counts, axis=0, keepdims=True)
        end0 = numpy.where(counts[:, 0] > 0)[0][-1] + 1
        end1 = numpy.where(counts[:, 1] > 0)[0][-1] + 1
        fig, ax = plt.subplots(1, 1)
        ax.plot(numpy.arange(end0), counts[:end0, 0], color='red')
        ax.plot(numpy.arange(end1), counts[:end1, 1], color='blue')
        plt.savefig('test_kest.pdf')
        plt.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()