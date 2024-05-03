#!/usr/bin/env python

import sys
import multiprocessing

import numpy
import HSMM
import matplotlib.pyplot as plt
import scipy.stats

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


    #     counts = numpy.zeros((2, 2), numpy.float64)
    #     ip = numpy.zeros(2, numpy.float64)
    #     tm = numpy.zeros((2, 2), numpy.float64)
    #     for i in range(len(seqs)):
    #         for j in range(2):
    #             where = numpy.where(states[i] == j)[0]
    #             counts[j, 0] += numpy.sum(seqs[i][where])
    #             counts[j, 1] += where.shape[0]
    #             tm[j, j] += numpy.sum(numpy.logical_and(
    #                             states[i][:-1] == j,
    #                             states[i][1:] == j))
    #             tm[j, (j+1)%2] += numpy.sum(numpy.logical_and(
    #                             states[i][:-1] == j,
    #                             states[i][1:] == (j+1)%2))
    #         ip[states[i][0]] += 1
    #     ip /= numpy.sum(ip) / 100
    #     counts = counts[:, 0] / counts[:, 1] * 100
    #     tm /= numpy.sum(tm, axis=1, keepdims=True)
    #     print("Dist p", counts)
    #     print("Pi", ip)
    #     print("TM", tm)
    #     model.save('test_model.npz')

    # with HSMM.HMM(fname='test_model.npz', num_threads=8, seed=2001) as model:
    #     print(model)


    # TM = numpy.array([[0.8, 0.2], [0.2, 0.8]], numpy.float64)
    # param = [[{"n": 1, "p": 0.3}], [{"n": 1, "p": 0.6}]]
    # with HSMM.HMM(
    #         num_states=2,
    #         num_threads=1,
    #         distributions=["BI"],
    #         transition_matrix=TM,
    #         initial_probabilities=None,
    #         seed=2001) as model:
    #     model.set_dist_parameters(param)
    #     print(model)
    #     model.load_observations(seqs)
    #     model.train(maxIterations=20)
    #     print(model)



if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()