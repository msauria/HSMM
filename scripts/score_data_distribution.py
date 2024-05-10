#!/usr/bin/env python

import sys
import os
import multiprocessing

import numpy
import HSMM

def main():
    in_fname, out_fname = sys.argv[1:3]
    all_data = numpy.load(in_fname, mmap_mode='r')
    N = all_data.shape[0]//4
    obs_breaks = numpy.round(numpy.linspace(0, N, 11)).astype(numpy.int32)
    dist_names = list(HSMM.distributions.Distribution.valid_dists.keys())
    dist_names.sort()
    names = all_data.dtype.names[3:]
    if len(sys.argv) > 3:
        index = int(sys.argv[3])
        nIdx = index // len(dist_names)
        dIdx = index % len(dist_names)
        dist_names = [dist_names[dIdx]]
        names = [names[nIdx]]
    results = []
    for name in names:
        seqs = []
        for i in range(obs_breaks.shape[0] - 1):
            s, e = obs_breaks[i:i+2]
            seqs.append(all_data[name][s:e].reshape(-1, 1))
        for dname in dist_names:
            with HSMM.HSMM(
                    num_states=3,
                    num_threads=10,
                    distributions=[dname],
                    transition_matrix=None,
                    initial_probabilities=None,
                    seed=2001) as model:
                model.load_observations(seqs)
                model.train(maxIterations=39, update_topology=10)
                results.append((name, dname, model.likelihood, model.AIC, model.BIC))
                print(model)

    fs = open(out_fname, 'w')
    fs.write(f"Dataset\tDistribution\t-LogLikelihood\tAIC\tBIC\n")
    for line in results:
        fs.write("\t".join([str(x) for x in line]) + "\n")
    fs.close()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()