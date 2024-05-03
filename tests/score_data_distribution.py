#!/usr/bin/env python

import sys
import os
import multiprocessing

import numpy
import HSMM

def main():
    in_fname, out_fname = sys.argv[1:3]
    all_data = numpy.load(in_fname)
    N = all_data.shape[0]
    obs_breaks = numpy.round(numpy.linspace(0, N, 11)).astype(numpy.int32)
    seqs = []
    for i in range(obs_breaks.shape[0] - 1):
        s, e = obs_breaks[i:i+2]
        seqs.append(all_data['count'][s:e].reshape(-1, 1))

    dist_names = list(HSMM.distributions.Distribution.valid_dists.keys())
    dist_names.sort()

    for name in dist_names:
        with HSMM.HMM(
                num_states=3,
                num_threads=5,
                distributions=[name],
                transition_matrix=None,
                initial_probabilities=None,
                seed=2001) as model:
            model.load_observations(seqs)
            model.cluster_observations(set_params=True)
            model.train(maxIterations=100)
            # print(model)

        if os.path.exists(out_fname):
            fs = open(out_fname, 'a')
        else:
            fs = open(out_fname, 'w')
            fs.write(f"Dataset\tDistribution\tAIC\tBIC\n")
        fs.write(f"{in_fname.split('/')[-1]}\t{name}\t{model.AIC}\t{model.BIC}\n")
        fs.close()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()