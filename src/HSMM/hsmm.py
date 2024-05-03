#!/usr/bin/env python

import sys
import multiprocessing
from copy import deepcopy

import numpy as np

from .hmm import HMM
from .distributions import Distribution


class HSMM(HMM):
    name = "HSMM"

    def _post_enter_actions(self):
        self.original_num_states = self.num_states
        if getattr(self, 'dwells', None) is None:
            self.make_shared_array('dwells', (self.num_states,), np.int32)
            self.dwells[:] = 1
            d_trans = self.transition_matrix
            d_init = self.initial_probabilities
            d_dists = self.distributions
        else:
            self.num_states = np.sum(self.dwells)
            self.make_shared_array('dwells', (self.original_num_states,), np.int32, self.dwells)
        d_trans, d_init, d_dists = self.convert_nodwell_to_dwell(self.dwells)
        self.make_shared_array('sizes', (6,), np.int64,
                               [self.num_states, self.num_distributions, 0, 0, 0,
                                self.original_num_states])
        self.make_shared_array('transition_matrix',
                               (self.num_states, self.num_states), np.float64,
                                d_trans)
        self.make_shared_array('initial_probabilities',
                               (self.num_states,), np.float64,
                               d_init)
        self.distributions = d_dists
        return self

    def print_model(self):
        output = [f"{self.name} with {self.original_num_states} states and {self.num_distributions} distribution(s)\n"]
        nd_trans, nd_init, nd_dists = self.convert_dwell_to_nodwell()
        output += self.print_states(nd_dists)
        output += self.print_transitions(nd_trans)
        output += self.print_initprobs(nd_init)
        return output

    def print_states(self, dists):
        num_states = len(dists)
        num_dists = len(dists[0])
        output = []
        for i in range(num_states):
            output.append(f"  State {i} - dwell {self.dwells[i]}")
            for j in range(num_dists):
                tmp = []
                for k, v in dists[i][j].params.items():
                    if v == np.round(v):
                        tmp.append(f"{k}: {v:d}")
                    else:
                        tmp.append(f"{k}: {v:0.2e}")
                tmp = "  ".join(tmp)
                output.append(f"    Distribution {j}: {dists[i][j].name} - {tmp}")
        return output

    def convert_dwell_to_nodwell(self):
        dwell_indices = np.r_[0, np.cumsum(self.dwells)]
        nd_trans = np.zeros((self.original_num_states, self.original_num_states), np.float64)
        nd_init = np.zeros(self.original_num_states, np.float64)
        distributions = []
        for i in range(self.original_num_states):
            nd_trans[i, :] = self.transition_matrix[dwell_indices[i+1]-1, dwell_indices[:-1]]
            nd_trans[i, i] = self.transition_matrix[dwell_indices[i], dwell_indices[i]]
            distributions.append(deepcopy(self.distributions[dwell_indices[i]]))
            nd_init[i] = self.initial_probabilities[dwell_indices[i]]
        return nd_trans, nd_init, distributions

    def convert_nodwell_to_dwell(self, dwells, transition_matrix=None, initial_probabilities=None, distributions=None):
        if transition_matrix is None:
            transition_matrix = self.transition_matrix
        transition_matrix = np.exp(transition_matrix)
        if initial_probabilities is None:
            initial_probabilities = self.initial_probabilities
        if distributions is None:
            distributions = self.distributions
        stateN = np.sum(dwells)
        dwell_indices = np.r_[0, np.cumsum(dwells)]
        d_trans = np.zeros((stateN, stateN), np.float64)
        d_init = np.full((stateN), -np.inf, np.float64)
        d_dists = []
        for i in range(self.original_num_states):
            s, e = dwell_indices[i:i+2]
            d_init[s] = initial_probabilities[i]
            d_dists.append(deepcopy(distributions[i]))
            p = transition_matrix[i, i]
            exit_probs = np.r_[transition_matrix[i, :i], 0, transition_matrix[i, i+1:]]
            exit_probs /= np.sum(exit_probs)
            NP = 1 - p
            for j in range(s, e):
                if j > s:
                    d_dists.append(d_dists[s])
                d_trans[j, j] = p
                nP = NP ** (e - j)
                d_trans[j, dwell_indices[:i]] = nP * exit_probs[:i]
                d_trans[j, dwell_indices[i + 1:-1]] = nP * exit_probs[i + 1:]
                prev = nP
                for k in range(j + 1, e)[::-1]:
                    nP = NP ** (k - j)
                    d_trans[j, k] = nP - prev
                    prev = nP
        d_trans = self.to_log(d_trans / np.sum(d_trans, axis=1, keepdims=True))
        return d_trans, d_init, d_dists

    def set_dwell_times(self, dwell_times):
        if (not isinstance(dwell_times, list) 
            and not isinstance(dwell_times, np.ndarray) 
            and len(dwell_times) != self.original_num_states):
            raise ValueError("Dwell times must be a list or int array of length equal to the number of states")
        try:
            dwells = np.array(dwell_times, np.int32)
        except:
            raise ValueError("Dwell times must be integers")
        if np.amin(dwells) <= 0:
            raise ValueError("Dwell times must be one or greater")
        if np.sum(dwells == self.dwells) == self.dwells.shape[0]:
            return
        nd_trans, nd_init, nd_dists = self.convert_dwell_to_nodwell()
        d_trans, d_init, d_dists = self.convert_nodwell_to_dwell(
            dwell_times, nd_trans, nd_init, nd_dists)

        self.dwells[:] = dwell_times
        self.num_states = np.sum(self.dwells)
        self.sizes[0] = self.num_states
        self.make_shared_array('transition_matrix',
                               (self.num_states, self.num_states), np.float64,
                               d_trans)
        self.make_shared_array('initial_probabilities',
                               (self.num_states,), np.float64,
                               d_init)
        self.distributions = d_dists
        return

    def save(self, fname):
        nd_trans, nd_init, nd_dists = self.convert_dwell_to_nodwell()
        super().save(fname, transition_matrix=nd_trans,
                     initial_probabilities=nd_init, distributions=nd_dists,
                     dwells=self.dwells)
        return





