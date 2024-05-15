#!/usr/bin/env python

import sys
import multiprocessing

import numpy
import HSMM
import matplotlib.pyplot as plt
import scipy.stats
from scipy.special import logsumexp

def main():
    RNG = numpy.random.default_rng(34584)
    stateN = 5
    distN = 3
    TM = numpy.zeros((stateN, stateN), numpy.float64)
    popN = 256
    p_bounds = numpy.linspace(0, 1, stateN + 2)
    params = []
    dwell_times = []
    for i in range(stateN):
        self_prob = RNG.random() * 0.2 + 0.75
        probs = RNG.random(stateN - 1)
        probs /= numpy.sum(probs)
        probs *= (1 - self_prob)
        TM[i, i] = self_prob
        TM[i, :i] = probs[:i]
        TM[i, i+1:] = probs[i:]
        params.append([])
        for j in range(distN):
            x = RNG.choice(p_bounds.shape[0] - 1)
            s, e = p_bounds[x:x+2]
            span = e - s
            s += span * 0.05
            e -= span * 0.05
            p = RNG.random() * (e - s) + s
            params[-1].append({'n':popN, 'p': p})
        dwell_times.append(RNG.poisson(1) + 1)
    IP = RNG.random(stateN)
    IP /= numpy.sum(IP)

    with HSMM.HSMM(
            num_states=stateN,
            num_threads=11,
            distributions=["BI" for x in range(distN)],
            transition_matrix=TM,
            initial_probabilities=IP,
            seed=2001) as model:
        model.set_dist_parameters(params)
        model.set_dwell_times(dwell_times)
        model.save('test_model.npz')

    with HSMM.HSMM(
            fname='test_model.npz',
            seed=2001) as model:
        print(model)
        seqs, states = model.generate_sequences(50, 5000)
        dwells = [[] for x in range(stateN)]
        dwell_indices = numpy.cumsum(model.dwells)
        for i, seq in enumerate(seqs):
            relabeled = numpy.searchsorted(dwell_indices, states[i], side='right') 
            breaks = numpy.r_[0, numpy.where(numpy.diff(relabeled))[0] + 1, seq.shape[0]]
            for j in range(breaks.shape[0] - 1):
                s, e = breaks[j:j+2]
                dwells[relabeled[s]].append(e - s)
        fig, ax = plt.subplots(1,1)
        cmap = plt.get_cmap('Set1')
        for i in range(stateN):
            Y = numpy.array(dwells[i]) - 1
            maxval = numpy.amax(Y)
            Y = numpy.bincount(Y)/Y.shape[0]
            ax.plot(numpy.arange(maxval + 1), Y, color=cmap(i))
            ax.plot(numpy.arange(maxval), scipy.stats.nbinom.pmf(
                numpy.arange(maxval), dwell_times[i], 1 - TM[i,i]),
                color=cmap(i), linestyle='dashed')
        plt.savefig('test_dwells.pdf')
        plt.close()
        model.load_observations(seqs)
        model.train(maxIterations=1, update=False)

    with HSMM.HSMM(
            num_states=stateN,
            num_threads=11,
            distributions=["BI" for x in range(distN)],
            transition_matrix=None,
            initial_probabilities=None,
            seed=2001) as model:
        model.load_observations(seqs)
        model.cluster_observations()
        model.set_dwell_times(dwell_times)
        model.train(maxIterations=11, update_topology=20)
        counts = model.maximize_dwell_times().astype(numpy.float64)
        print(model)
        counts /= numpy.sum(counts, axis=1, keepdims=True)
        nd_TM, _, _ = model.convert_dwell_to_nodwell()
        nd_TM = numpy.exp(nd_TM)
        cmap = plt.get_cmap('Set1')
        fig, ax = plt.subplots(1, 1)
        for i in range(stateN):
            end = numpy.where(counts[i, :] > 0)[0][-1] + 1
            ax.plot(numpy.arange(end), counts[i, :end], color=cmap(i))
            ax.plot(numpy.arange(end), scipy.stats.nbinom.pmf(
                numpy.arange(end), model.dwells[i], 1 - nd_TM[i,i]),
                color=cmap(i), linestyle='dashed')
        plt.savefig('test_kest.pdf')
        plt.close()
        remapping = numpy.repeat(numpy.arange(stateN), dwell_times)
        new_states, _ = model.viterbi()
        states = remapping[numpy.concatenate(states, axis=0)]
        new_states = remapping[numpy.concatenate(new_states, axis=0)]
        counts = numpy.bincount(states * stateN + new_states, minlength=stateN**2).reshape(stateN, stateN)
        print(counts)


if __name__ == "__main__":
    main()





