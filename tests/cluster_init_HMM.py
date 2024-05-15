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
    IP = RNG.random(stateN)
    IP /= numpy.sum(IP)

    with HSMM.HMM(
            num_states=stateN,
            num_threads=11,
            distributions=["BI", "BI", "BI"],
            transition_matrix=TM,
            initial_probabilities=IP,
            seed=2001) as model:
        model.set_dist_parameters(params)
        model.save('test_model.npz')

    with HSMM.HMM(
            fname='test_model.npz',
            seed=2001) as model:
        print(model)
        seqs, states = model.generate_sequences(50, 10000)
        model.load_observations(seqs)
        model.train(maxIterations=1, update=False)

    with HSMM.HMM(
            num_states=stateN,
            num_threads=11,
            distributions=["BI", "BI", "BI"],
            transition_matrix=None,
            initial_probabilities=None,
            seed=2001) as model:
        model.load_observations(seqs)
        model.cluster_observations()
        model.train(maxIterations=11, update_topology=3)
        print(model)


if __name__ == "__main__":
    main()





