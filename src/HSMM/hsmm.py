#!/usr/bin/env python

import sys
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
from copy import deepcopy
import warnings

import numpy as np
import scipy.stats
from scipy.special import logsumexp

from .hmm import HMM
from .distributions import Distribution


class HSMM(HMM):
    name = "HSMM"
    maxdwell = 10

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
                        tmp.append(f"{k}: {int(v):d}")
                    else:
                        tmp.append(f"{k}: {v:0.2e}")
                tmp = "  ".join(tmp)
                output.append(f"    Distribution {j}: {dists[i][j].name} - {tmp}")
        return output

    def num_free_parameters(self):
        s = 0
        for i in range(self.num_distributions):
            s += len(self.distributions[0][i].params)
        p = self.original_num_states * (self.original_num_states + 2 + s) - 1
        return p

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

    def convert_nodwell_to_dwell(self, dwells, transition_matrix=None, initial_probabilities=None, distributions=None, log=True):
        if transition_matrix is None:
            transition_matrix = self.transition_matrix
        if log:
            transition_matrix = np.exp(transition_matrix)
        if initial_probabilities is None:
            initial_probabilities = self.initial_probabilities
        if distributions is None:
            distributions = self.distributions
        stateN = np.sum(dwells)
        dwell_indices = np.r_[0, np.cumsum(dwells)]
        d_trans = np.zeros((stateN, stateN), np.float64)
        if log:
            d_init = np.full((stateN), -np.inf, np.float64)
        else:
            d_init = np.zeros(stateN, np.float64)
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
        if log:
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
        prev_stateN = self.num_states
        nd_trans, nd_init, nd_dists = self.convert_dwell_to_nodwell()
        d_trans, d_init, d_dists = self.convert_nodwell_to_dwell(
            dwells, nd_trans, nd_init, nd_dists)

        self.dwells[:] = dwells
        self.num_states = np.sum(self.dwells)
        self.sizes[0] = self.num_states
        self.make_shared_array('transition_matrix',
                               (self.num_states, self.num_states), np.float64,
                               d_trans)
        self.make_shared_array('initial_probabilities',
                               (self.num_states,), np.float64,
                               d_init)
        self.distributions = d_dists
        if self.num_states != prev_stateN and self.num_maxes > 0:
            self.make_shared_array("probs", (self.num_maxes, self.num_states),
                                   np.float64)
            self.make_shared_array("emissions", (self.num_obs, self.num_states),
                                   np.float64)
            self.make_shared_array("alpha", (self.num_obs, self.num_states),
                                   np.float64)
            self.make_shared_array("beta", (self.num_obs, self.num_states),
                                   np.float64)
            self.make_shared_array("lngamma", (self.num_obs, self.num_states),
                                   np.float64)
            self.make_shared_array("gamma", (self.num_obs, self.num_states),
                                   np.float64)
        return

    def maximization_step(self, iteration, **kwargs):
        if 'update_topology' in kwargs:
            update_topology = max(1, int(kwargs['update_topology']))
        else:
            update_topology = 1
        dwell_indices = np.r_[0, np.cumsum(self.dwells)]
        futures = []
        nd_dists = []
        for i in range(self.original_num_states):
            s, e = dwell_indices[i:i+2]
            nd_dists.append(self.distributions[s])
            for j in range(self.num_distributions):
                futures.append(self.pool.apply_async(
                    self._maximize_emissions_thread,
                    args=(s, e, i, j, self.distributions[s][j], self.smm_map)))
        for f in futures:
            sIdx, dIdx, params = f.get()
            nd_dists[sIdx][dIdx].set_parameters(**params)

        futures = []
        for i in range(self.thread_seq_indices.shape[0] - 1):
            s, e = self.thread_seq_indices[i:i+2]
            futures.append(self.pool.apply_async(
                self._maximize_transitions_thread,
                args=(s, e, self.smm_map)))
        initial_probabilities = np.zeros(self.original_num_states, np.float64)
        transition_matrix = np.zeros((self.original_num_states,
                                      self.original_num_states), np.float64)
        for f in futures:
            tmp = f.get()
            initial_probabilities += tmp[0]
            transition_matrix += tmp[1]
        initial_probabilities /= np.sum(initial_probabilities)
        transition_matrix /= np.sum(transition_matrix, axis=1, keepdims=True)
        d_TM, d_IP, d_dists = self.convert_nodwell_to_dwell(
            self.dwells, transition_matrix=transition_matrix,
            initial_probabilities=initial_probabilities,
            distributions=nd_dists, log=False)
        self.initial_probabilities[:] = self.to_log(d_IP)
        self.transition_matrix[:, :] = self.to_log(d_TM)
        self.distributions = d_dists
        if iteration % update_topology == 0:
            self.maximize_dwell_times()
        return

    @classmethod
    def _maximize_emissions_thread(cls, start_sIdx, end_sIdx, sIdx, dIdx, distribution, smm_map):
        views = []
        views.append(SharedMemory(smm_map['sizes']))
        sizes = np.ndarray(6, np.int64, buffer=views[-1].buf)
        stateN, distN, obsN, seqN = sizes[:4]
        views.append(SharedMemory(smm_map['obs']))
        obs = np.ndarray((obsN, distN), np.int32, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['gamma']))
        gamma = np.ndarray((obsN, stateN), np.float64, buffer=views[-1].buf)
        params = distribution.optimize_parameters(
            obs[:, dIdx], np.sum(gamma[:, start_sIdx:end_sIdx], axis=1))
        for view in views:
            view.close()
        return sIdx, dIdx, params

    @classmethod
    def _maximize_transitions_thread(cls, start, end, smm_map):
        views = []
        views.append(SharedMemory(smm_map['sizes']))
        sizes = np.ndarray(6, np.int64, buffer=views[-1].buf)
        stateN, distN, obsN, seqN, _, origstateN = sizes
        views.append(SharedMemory(smm_map['dwells']))
        dwells = np.ndarray((origstateN), np.int32, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['emissions']))
        emissions = np.ndarray((obsN, stateN), np.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['alpha']))
        alpha = np.ndarray((obsN, stateN), np.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['beta']))
        beta = np.ndarray((obsN, stateN), np.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['lngamma']))
        lngamma = np.ndarray((obsN, stateN), np.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['gamma']))
        gamma = np.ndarray((obsN, stateN), np.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['transition_matrix']))
        transitions = np.ndarray((stateN, stateN), np.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['obs_indices']))
        obs_indices = np.ndarray((seqN + 1), np.int64, buffer=views[-1].buf)
        dwell_indices = np.r_[0, np.cumsum(dwells)]

        s = obs_indices[start]
        e = obs_indices[end]
        new_init_probs = logsumexp(lngamma[s:e, :] - logsumexp(
            lngamma[s:e, :], axis=1, keepdims=True), axis=0)
        new_trans = np.full((e - s, stateN, stateN), -np.inf, np.float64)

        for i in range(start, end):
            s, e = obs_indices[i:i+2]
            tmp = (alpha[s:e-1, :, np.newaxis] + transitions[np.newaxis, :, :] +
                   emissions[s+1:e, np.newaxis, :] + beta[s+1:e, np.newaxis, :])
            new_trans[i - start, :, :] = logsumexp(tmp - logsumexp(
                tmp.reshape(e - s - 1, -1), axis=1).reshape(-1, 1, 1), axis=0)
        new_trans = logsumexp(new_trans, axis=0)

        new_trans = np.exp(new_trans)
        nd_trans = np.zeros((origstateN, origstateN), np.float64)
        for i in range(origstateN):
            s, e = dwell_indices[i:i+2]
            nd_trans[i, i] = np.sum(new_trans[np.arange(s, e), np.arange(s, e)])
            nd_trans[i, :i] = np.sum(new_trans[s:e, dwell_indices[:i]], axis=0)
            nd_trans[i, i+1:] = np.sum(new_trans[s:e, dwell_indices[i+1:-1]], axis=0)
        new_init_probs = np.exp(new_init_probs)
        nd_init_probs = new_init_probs[dwell_indices[:-1]]

        for view in views:
            view.close()
        return nd_init_probs, nd_trans

    def maximize_dwell_times(self):
        states, _ = self.viterbi()
        states = np.concatenate(states, axis=0)
        remapping = np.searchsorted(np.cumsum(self.dwells), np.arange(self.num_states),
                                    side='right')
        self.make_shared_array('states', (self.num_obs,), np.int32,
                               remapping[states])
        futures = []
        maxdwell = 1000
        for i in range(self.thread_seq_indices.shape[0] - 1):
            s, e = self.thread_seq_indices[i:i+2]
            futures.append(self.pool.apply_async(
                self._calculate_dwell_count_thread,
                args=(s, e, maxdwell, self.smm_map)))
        dwell_counts = np.zeros((self.original_num_states, maxdwell), np.int32)
        for f in futures:
            dwell_counts += f.get()
        nd_trans, _, _ = self.convert_dwell_to_nodwell()
        nd_trans = np.exp(nd_trans)

        futures = []
        for i in range(self.original_num_states):
            futures.append(self.pool.apply_async(
                self._calculate_dwell_prob_thread,
                args=(i, dwell_counts[i, :], 1 - nd_trans[i, i])))
        new_dwells = np.zeros(self.original_num_states, np.int32)
        for f in futures:
            sIdx, params = f.get()
            new_dwells[sIdx] = params['n']
            nd_trans[sIdx, sIdx] = 0
            nd_trans[sIdx, :] *= params['p'] / np.sum(nd_trans[sIdx, :])
            nd_trans[sIdx, sIdx] = 1 - params['p']
        d_trans, _, _ = self.convert_nodwell_to_dwell(
            self.dwells, transition_matrix=self.to_log(nd_trans))
        self.transition_matrix[:, :] = d_trans
        self.set_dwell_times(new_dwells)
        self.delete_shared_array('states')
        return dwell_counts

    @classmethod
    def _calculate_dwell_count_thread(cls, start, end, maxdwell, smm_map):
        views = []
        views.append(SharedMemory(smm_map['sizes']))
        sizes = np.ndarray(6, np.int64, buffer=views[-1].buf)
        stateN, distN, obsN, seqN, _, origstateN = sizes
        views.append(SharedMemory(smm_map['states']))
        states = np.ndarray((obsN), np.int32, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['obs_indices']))
        obs_indices = np.ndarray((seqN + 1), np.int64, buffer=views[-1].buf)
        dwell_count = np.zeros((origstateN, maxdwell), np.int32)
        for i in range(start, end):
            s, e = obs_indices[i:i+2]
            breaks = np.r_[0, np.where(np.diff(states[s:e]))[0] + 1,
                           e - s]
            for j in range(breaks.shape[0] - 1):
                bs, be = breaks[j:j+2]
                index = states[bs + s]
                span = be - bs - 1
                if span >= maxdwell:
                    continue
                dwell_count[index, span] += 1

        for view in views:
            view.close()
        return dwell_count

    @classmethod
    def _calculate_dwell_prob_thread(cls, sIdx, counts, p):
        warnings.filterwarnings("ignore")
        X = np.repeat(np.arange(counts.shape[0]), counts)
        bounds = {'p': (0, 1), 'n': (1, cls.maxdwell)}
        est_params = {'p': p, 'n': 1}
        fit = scipy.stats.fit(scipy.stats.nbinom, X, bounds, guess=est_params)
        new_params = {k: v for k, v in fit.params._asdict().items() if k in est_params}
        return sIdx, new_params




    def save(self, fname):
        nd_trans, nd_init, nd_dists = self.convert_dwell_to_nodwell()
        super().save(fname, transition_matrix=nd_trans,
                     initial_probabilities=nd_init, distributions=nd_dists,
                     dwells=self.dwells)
        return





