#!/usr/bin/env python

import sys
import math
import multiprocessing
import multiprocessing.managers
from multiprocessing.resource_tracker import unregister
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from .distributions import Distribution


class HMM():
    minfloat64 = np.nextafter(0, 1)
    name = "HMM"

    def __init__(
            self,
            num_states=2,
            num_threads=1,
            distributions=[],
            transition_matrix=None,
            initial_probabilities=None,
            seed=None,
            fname=None):
        self.RNG = np.random.default_rng(seed)
        self.num_threads = max(1, int(num_threads))
        if fname is not None:
            self.load(fname)
            return
        self.num_states = int(num_states)
        if len(distributions) < 1:
            raise ValueError("The number of distributions must be >= 1")
        self.num_distributions = len(distributions)
        self.distributions = [[] for x in range(self.num_states)]
        for D in distributions:
            if D not in Distribution.valid_dists:
                raise ValueError(f"{D} is not a valid distribution")
            for i in range(self.num_states):
                self.distributions[i].append(Distribution(D, self.RNG.spawn(1)[0]))
        if transition_matrix is None:
            self.transition_matrix = np.full((self.num_states, num_states),
                                             np.log(1/self.num_states), np.float64)
        else:
            if (not isinstance(transition_matrix, np.ndarray) or
                transition_matrix.shape != (self.num_states, self.num_states)):
                raise ValueError(f"The transition_matrix must be np.matrix of shape (num_states, num_states)")
            self.transition_matrix = self.to_log(transition_matrix /
                                                 np.sum(transition_matrix,
                                                        axis=1, keepdims=True))
        if initial_probabilities is None:
            self.initial_probabilities = np.full(self.num_states,
                                                 np.log(1/self.num_states),
                                                 np.float64)
        else:
            if (not isinstance(transition_matrix, np.ndarray) or
                initial_probabilities.shape != (self.num_states,)):
                raise ValueError(f"The initial_probabilities must be a np vector of shape (num_states,)")
            self.initial_probabilities = self.to_log(initial_probabilities /
                                                     np.sum(initial_probabilities))
        self.num_obs = 0
        self.num_seqs = 0
        self.num_maxes = 0
        self.likelihood = 0
        return

    def __enter__(self):
        self.smm_map = {}
        self.views = {}
        self.pool = multiprocessing.Pool(self.num_threads)
        self._post_enter_actions()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.pool.close()
        self.pool.terminate()
        for view in self.views.values():
            view.close()
            view.unlink()
        # self.smm.shutdown()
        return

    def _post_enter_actions(self):
        self.make_shared_array('sizes', (6,), np.int64,
                               [self.num_states, self.num_distributions, 0, 0, 0, 0])
        self.make_shared_array('transition_matrix',
                               (self.num_states, self.num_states), np.float64,
                                self.transition_matrix)
        self.make_shared_array('initial_probabilities',
                               (self.num_states,), np.float64,
                               self.initial_probabilities)
        return

    def __str__(self):
        return "\n".join(self.print_model())

    def print_model(self):
        output = [f"{self.name} with {self.num_states} states and {self.num_distributions} distribution(s)\n"]
        output += self.print_states(self.distributions)
        output += self.print_transitions(self.transition_matrix)
        output += self.print_initprobs(self.initial_probabilities)
        return output

    def print_states(self, dists):
        num_states = len(dists)
        num_dists = len(dists[0])
        output = []
        for i in range(num_states):
            output.append(f"  State {i}")
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

    def print_transitions(self, transition_matrix):
        num_states = transition_matrix.shape[0]
        output = [f"\n  Transition Matrix"]
        TM = np.exp(transition_matrix)
        pad = len(str(num_states - 1)) + 1
        pad2 = max([len(f"{x:0.2e}") for x in TM.ravel()])
        tmp = []
        for i in range(num_states):
            p = pad2 - len(str(i))
            p1 = p // 2
            p0 = p - p1
            tmp.append(f"{' '*p0}{i}{' '*p1}")
        tmp = " | ".join(tmp)
        output.append(f"{' '*(pad + 4)} | {tmp}")
        for i in range(num_states):
            tmp = "-+-".join([f"{'-'*pad2}" for x in range(num_states)])
            output.append(f"{' ' * 4}{'-' * pad}-+-{tmp}")
            tmp = " | ".join([f"{x:0.2e}".rjust(pad2) for x in TM[i, :]])
            output.append(f"{' '*4}{str(i).rjust(pad, ' ')} | {tmp}")
        return output

    def print_initprobs(self, initial_probabilities):
        num_states = initial_probabilities.shape[0]
        output = [f"\n  Initial Probabilities"]
        IP = np.exp(initial_probabilities)
        pad = len(str(num_states - 1)) + 1
        pad2 = max([len(f"{x:0.2e}") for x in IP])
        tmp = []
        for i in range(num_states):
            p = pad2 - len(str(i))
            p1 = p // 2
            p0 = p - p1
            tmp.append(f"{' '*p0}{i}{' '*p1}")
        tmp = " | ".join(tmp)
        output.append(f"{' '*(pad + 4)} | {tmp}")
        tmp = "-+-".join([f"{'-'*pad2}" for x in range(num_states)])
        output.append(f"{' ' * 4}{'-' * pad}-+-{tmp}")
        tmp = " | ".join([f"{x:0.2e}".rjust(pad2) for x in IP])
        output.append(f"{' '*4}{str(i).rjust(pad, ' ')} | {tmp}")
        return output

    @classmethod
    def product(cls, X):
        prod = 1
        for x in X:
            prod *= x
        return prod

    def make_shared_array(self, name, shape, dtype, data=None):
        new_size = ((self.product(shape) * np.dtype(dtype).itemsize - 1) //
                    4096 + 1) * 4096
        if name in self.views and self.views[name].size == new_size:
            return getattr(self, name)
        if name in self.views:
            self.delete_shared_array(name)
        self.views[name] = SharedMemory(create=True, size=new_size)
        self.smm_map[name] = self.views[name].name
        new_data = np.ndarray(shape, dtype, buffer=self.views[name].buf)
        if data is not None:
            new_data[:] = data
        setattr(self, name, new_data)
        return new_data

    def delete_shared_array(self, name):
        self.views[name].close()
        self.views[name].unlink()
        del self.views[name]
        del self.smm_map[name]
        return

    @classmethod
    def to_log(cls, data):
        ldata = np.full(data.shape, -np.inf, data.dtype)
        where = np.where(data > 0)
        ldata[where] = np.log(data[where])
        return ldata

    def generate_sequences(self, num_sequences, lengths):
        if (not isinstance(lengths, int) and
            not isintance(lengths, np.ndarray) and
            not isintance(lenths, list)):
            raise ValueError("Lengths must be an int or list/array of length equal to the number of sequences")
        if not isinstance(num_sequences, int) or num_sequences <= 0:
            raise ValueError("Number of sequences must be a positive integer")
        if isinstance(lengths, int):
            lengths = np.full(num_sequences, lengths, np.int32)
        else:
            lengths = np.array(lengths, np.int32)
        futures = []
        indices = np.round(np.linspace(0, num_sequences,
                                             self.num_threads + 1)).astype(np.int32)
        for i in range(indices.shape[0] - 1):
            s, e = indices[i:i+2]
            futures.append(self.pool.apply_async(
                self._generate_sequence_thread,
                args=(s, e, lengths[s:e], self.distributions,
                      self.RNG.spawn(1)[0], self.smm_map)))
        obs = [None for x in range(num_sequences)]
        states = [None for x in range(num_sequences)]
        for f in futures:
            s, e, O, S = f.get()
            obs[s:e] = O
            states[s:e] = S
        return obs, states

    @classmethod
    def _generate_sequence_thread(cls, start, end, lengths, dists, RNG, smm_map):
        views = []
        views.append(SharedMemory(smm_map['sizes']))
        sizes = np.ndarray(6, np.int64, buffer=views[-1].buf)
        stateN, distN, obsN, seqN, maxN = sizes[:5]
        views.append(SharedMemory(smm_map['transition_matrix']))
        transitions = np.ndarray((stateN, stateN), np.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['initial_probabilities']))
        init_probs = np.ndarray((stateN,), np.float64, buffer=views[-1].buf)
        num_seqs = end - start
        all_obs = []
        all_states = []
        stateN = len(dists)
        distN = len(dists[0])
        expIP = np.exp(init_probs)
        expIP /= np.sum(expIP)
        expTM = np.exp(transitions)
        expTM /= np.sum(expTM, axis=1, keepdims=True)
        for i in range(num_seqs):
            states = np.zeros(lengths[i], np.int32)
            obs = np.zeros((lengths[i], distN), np.float64)
            states[0] = RNG.choice(stateN, p=expIP)
            for j in range(distN):
                obs[0, j] = dists[states[0]][j].rvs()
            for j in range(1, lengths[i]):
                states[j] = RNG.choice(stateN, p=expTM[states[j - 1], :])
                for k in range(distN):
                    obs[j, k] = dists[states[j]][k].rvs()
            all_obs.append(obs)
            all_states.append(states)
        for view in views:
            view.close()
        return start, end, all_obs, all_states

    def load_observations(self, observations, initial_states=None):
        if not isinstance(observations, list) or len(observations) == 0:
            raise ValueError("Observations must be a list of sequences")
        n = 0
        for O in observations:
            if (not isinstance(O, np.ndarray) or
                (O.shape[1] != self.num_distributions) or
                (not O.dtype in [np.int32, np.float64])):
                raise ValueError("Each sequence must be a integer np matrix of shape (X, num_distributions)")
            n += O.shape[0]
        self.num_seqs = len(observations)
        self.num_obs = n
        self.sizes[2] = self.num_obs
        self.sizes[3] = self.num_seqs
        self.make_shared_array('obs', (self.num_obs, self.num_distributions),
                               np.int32)
        self.make_shared_array('obs_indices', (self.num_seqs + 1,), np.int64)
        self.obs_indices[0] = 0
        for i, O in enumerate(observations):
            s = self.obs_indices[i]
            e = s + O.shape[0]
            self.obs_indices[i + 1] = e
            self.obs[s:e, :] = O
        self.make_shared_array('max_indices', (self.num_distributions + 1,), np.int64)
        self.max_indices[0] = 0
        for i in range(self.num_distributions):
            self.max_indices[i + 1] = self.max_indices[i] + np.amax(self.obs[:, i]) + 1
        self.num_maxes = self.max_indices[-1]
        self.sizes[4] = self.num_maxes
        self.make_thread_indices()
        self.set_dist_bounds()
        if not initial_states is None and not initial_states:
            if isinstance(initial_states, list):
                if len(initial_states) != self.num_seqs or sum([len(x) for x in initial_states]):
                    raise ValueError("The initial states do not match the observation num/lengths")
                initstates = np.concatenate(initial_states, axis=0)
            else:
                initstates = self.RNG.choice(self.num_states, size=self.num_obs)
            self.set_dist_estimates(initstates)
        return

    def make_thread_indices(self):
        self.thread_obs_indices = np.round(np.linspace(
            0, self.num_obs, self.num_threads * 3 + 1)).astype(np.int64)
        self.thread_seq_indices = np.searchsorted(
            self.obs_indices[1:], self.thread_obs_indices, side='right')
        self.thread_seq_indices[1:-1] += np.round(
            (self.thread_obs_indices[1:-1] -
             self.obs_indices[self.thread_seq_indices[1:-1]]) /
            (self.obs_indices[self.thread_seq_indices[1:-1] + 1] -
             self.obs_indices[self.thread_seq_indices[1:-1]])).astype(np.int64)
        self.thread_obs_indices = np.r_[self.thread_obs_indices[np.where(np.diff(
            self.thread_obs_indices) > 0)], self.thread_obs_indices[-1]].astype(np.int64)
        self.thread_seq_indices = np.r_[self.thread_seq_indices[np.where(np.diff(
            self.thread_seq_indices) > 0)], self.thread_seq_indices[-1]].astype(np.int64)
        return

    def cluster_observations(self, set_params=True):
        if self.num_seqs == 0:
            raise RuntimeError("Observations must be loaded before running clustering")
        # Can replace random_state with self.RNG once scikit-learn upgrades from RandomState
        kmeans = KMeans(n_clusters=self.num_states, n_init='auto',
                        random_state=np.random.RandomState(self.RNG.bit_generator))
        states = kmeans.fit_predict(self.obs)
        if set_params:
            self.set_dist_estimates(states)
        return states, kmeans.cluster_centers_

    def set_dist_bounds(self):
        for i in range(self.num_states):
            for j in range(self.num_distributions):
                self.distributions[i][j].get_bounds(self.obs)
        return

    def set_dist_parameters(self, params):
        for i in range(self.num_states):
            for j in range(self.num_distributions):
                self.distributions[i][j].set_parameters(**params[i][j])
        return

    def set_dist_estimates(self, states):
        for i in range(self.num_states):
            where = np.where(states == i)[0]
            for j in range(self.num_distributions):
                self.distributions[i][j].get_bounds(self.obs)
                self.distributions[i][j].estimate_params(self.obs[where, j])
                self.distributions[i][j].optimize_parameters(self.obs[where, j])
        return

    def viterbi(self):
        if self.num_seqs == 0:
            raise RuntimeError("Observations must be loaded before running viterbi")
        self.make_shared_array("probs", (self.num_maxes, self.num_states),
                               np.float64)
        self.make_shared_array("emissions", (self.num_obs, self.num_states),
                               np.float64)
        self.calculate_probabilities()
        self.calculate_emissions()
        return self.calculate_paths()

    def calculate_paths(self):
        futures = []
        for i in range(self.thread_seq_indices.shape[0] - 1):
            s, e = self.thread_seq_indices[i:i+2]
            futures.append(self.pool.apply_async(
                self._calculate_path_thread,
                args=(s, e, self.smm_map)))
        states = [None for x in range(self.num_seqs)]
        scores = np.zeros(self.num_seqs, np.float64)
        for f in futures:
            s, e, P, S = f.get()
            states[s:e] = P
            scores[s:e] = S
        return states, scores

    @classmethod
    def _calculate_path_thread(cls, start, end, smm_map):
        views = []
        views.append(SharedMemory(smm_map['sizes']))
        sizes = np.ndarray(6, np.int64, buffer=views[-1].buf)
        stateN, distN, obsN, seqN, maxN = sizes[:5]
        views.append(SharedMemory(smm_map['emissions']))
        emissions = np.ndarray((obsN, stateN), np.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['transition_matrix']))
        transitions = np.ndarray((stateN, stateN), np.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['initial_probabilities']))
        init_probs = np.ndarray((stateN,), np.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['obs_indices']))
        obs_indices = np.ndarray((seqN + 1), np.int64, buffer=views[-1].buf)

        all_states = []
        all_scores = np.zeros(end - start, np.float64)
        for i in range(start, end):
            s, e = obs_indices[i:i+2]
            forward = np.zeros((e - s, stateN), np.float64)
            paths = np.full((e - s, stateN), -1, np.int32)
            forward[0, :] = init_probs + emissions[s, :]
            for j in range(s + 1, e):
                k = j - s
                trans = forward[k - 1, :, np.newaxis] + transitions
                best = np.argmax(trans, axis=0)
                paths[k - 1, :] = best
                forward[k, :] = trans[best, np.arange(stateN)] + emissions[j, :]
            paths[-1, :] = np.argmax(forward[-1, :])
            states = np.zeros(e - s, np.int32)
            states[-1] = paths[-1, 0]
            all_scores = forward[-1, states[-1]]
            for j in range(e - s - 1)[::-1]:
                states[j] = paths[j, states[j + 1]]
            all_states.append(states)

        for view in views:
            view.close()
        return start, end, all_states, all_scores

    def train(self, epsilon=1e-1, maxIterations=0, update=True, **kwargs):
        if self.num_seqs == 0:
            raise RuntimeError("Observations must be loaded before training")
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
        iteration = 0
        oldLikelihood = np.inf
        newLikelihood = np.inf
        while True:
            iteration += 1
            oldLikelihood = newLikelihood
            newLikelihood = self.training_iteration(iteration, update, **kwargs)
            self.likelihood = newLikelihood
            if maxIterations > 0 and iteration == maxIterations:
                break
            # if newLikelihood > oldLikelihood:
            #     print(f"Training stopped: new likelihood smaller than old likelihood",
            #           file=sys.stderr)
            #     break
            if iteration % 10 == 0:
                print(f"Performed iteration {iteration} with loglikelihood {newLikelihood}",
                      file=sys.stderr)
            # if math.fabs(oldLikelihood - newLikelihood) <= epsilon:
            #     break
        print(f"Training successfully completed after {iteration} iterations with loglikelihood {newLikelihood}",
              file=sys.stderr)
        p = self.num_free_parameters()
        self.AIC = 2 * newLikelihood + 2 * p
        self.BIC = 2 * newLikelihood + p * self.num_obs
        print(f"AIC: {self.AIC}\nBIC: {self.BIC}", file=sys.stderr)
        return

    def training_iteration(self, iteration, update, **kwargs):
        self.calculate_probabilities()
        self.calculate_emissions()
        newLikelihood = self.calculate_forwardpass()
        if iteration == 1:
            print(f"Initial loglikelihood {newLikelihood}", file=sys.stderr)
        self.calculate_backwardpass()
        self.expectation_step()
        if update:
            self.maximization_step(iteration=iteration, **kwargs)
        return newLikelihood


    def num_free_parameters(self):
        s = 0
        for i in range(self.num_distributions):
            s += len(self.distributions[0][i].params)
        p = self.num_states * (self.num_states + 1 + s) - 1
        return p

    def calculate_probabilities(self):
        futures = []
        for sIdx in range(self.num_states):
            for dIdx in range(self.num_distributions):
                futures.append(self.pool.apply_async(
                    self._probability_thread,
                    args=(sIdx, dIdx, self.distributions[sIdx][dIdx], self.smm_map)))
        for f in futures:
            _ = f.get()
        return

    @classmethod
    def _probability_thread(cls, sIdx, dIdx, distribution, smm_map):
        views = []
        views.append(SharedMemory(smm_map['sizes']))
        sizes = np.ndarray(6, np.int64, buffer=views[-1].buf)
        stateN, distN, obsN, seqN, maxN = sizes[:5]
        views.append(SharedMemory(smm_map['obs']))
        obs = np.ndarray((obsN, distN), np.int32, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['probs']))
        probs = np.ndarray((maxN, stateN), np.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['max_indices']))
        max_indices = np.ndarray((distN + 1,), np.int64, buffer=views[-1].buf)

        s, e = max_indices[dIdx:dIdx+2]
        probs[s:e, sIdx] = distribution.logpmf(np.arange(e - s))
        for view in views:
            view.close()
        return

    def calculate_emissions(self):
        futures = []
        for i in range(self.thread_obs_indices.shape[0] - 1):
            s, e = self.thread_obs_indices[i:i+2]
            futures.append(self.pool.apply_async(
                self._emission_thread,
                args=(s, e, self.smm_map)))
        for f in futures:
            _ = f.get()
        return

    @classmethod
    def _emission_thread(cls, start, end, smm_map):
        views = []
        views.append(SharedMemory(smm_map['sizes']))
        sizes = np.ndarray(6, np.int64, buffer=views[-1].buf)
        stateN, distN, obsN, seqN, maxN = sizes[:5]
        views.append(SharedMemory(smm_map['obs']))
        obs = np.ndarray((obsN, distN), np.int32, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['probs']))
        probs = np.ndarray((maxN, stateN), np.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['max_indices']))
        max_indices = np.ndarray((distN + 1,), np.int64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['emissions']))
        emissions = np.ndarray((obsN, stateN), np.float64, buffer=views[-1].buf)

        for i in range(stateN):
            emissions[start:end, i] = 0
            for j in range(distN):
                s = max_indices[j]
                emissions[start:end, i] += probs[s + obs[start:end, j], i]

        for view in views:
            view.close()
        return

    def calculate_forwardpass(self):
        futures = []
        for i in range(self.thread_seq_indices.shape[0] - 1):
            s, e = self.thread_seq_indices[i:i+2]
            futures.append(self.pool.apply_async(
                self._forwardpass_thread,
                args=(s, e, self.smm_map)))
        likelihood = 0
        for f in futures:
            likelihood += f.get()
        return likelihood

    @classmethod
    def _forwardpass_thread(cls, start, end, smm_map):
        views = []
        views.append(SharedMemory(smm_map['sizes']))
        sizes = np.ndarray(6, np.int64, buffer=views[-1].buf)
        stateN, distN, obsN, seqN = sizes[:4]
        views.append(SharedMemory(smm_map['emissions']))
        emissions = np.ndarray((obsN, stateN), np.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['alpha']))
        alpha = np.ndarray((obsN, stateN), np.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['transition_matrix']))
        transitions = np.ndarray((stateN, stateN), np.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['initial_probabilities']))
        init_probs = np.ndarray((stateN,), np.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['obs_indices']))
        obs_indices = np.ndarray((seqN + 1), np.int64, buffer=views[-1].buf)

        likelihood = 0
        for i in range(start, end):
            s, e = obs_indices[i:i+2]
            alpha[s, :] = init_probs + emissions[s, :]
            for j in range(s + 1, e):
                alpha[j, :] = logsumexp(alpha[j - 1, :].reshape(-1, 1) +
                                        transitions, axis=0) + emissions[j, :]
            likelihood += logsumexp(alpha[e - 1, :])
        for view in views:
            view.close()
        return -likelihood

    def calculate_backwardpass(self):
        futures = []
        for i in range(self.thread_seq_indices.shape[0] - 1):
            s, e = self.thread_seq_indices[i:i+2]
            futures.append(self.pool.apply_async(
                self._backwardpass_thread,
                args=(s, e, self.smm_map)))
        for f in futures:
            _ = f.get()
        return

    @classmethod
    def _backwardpass_thread(cls, start, end, smm_map):
        views = []
        views.append(SharedMemory(smm_map['sizes']))
        sizes = np.ndarray(6, np.int64, buffer=views[-1].buf)
        stateN, distN, obsN, seqN = sizes[:4]
        views.append(SharedMemory(smm_map['emissions']))
        emissions = np.ndarray((obsN, stateN), np.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['beta']))
        beta = np.ndarray((obsN, stateN), np.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['transition_matrix']))
        transitions = np.ndarray((stateN, stateN), np.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['obs_indices']))
        obs_indices = np.ndarray((seqN + 1), np.int64, buffer=views[-1].buf)

        for i in range(start, end):
            s, e = obs_indices[i:i+2]
            beta[e - 1, :] = 0
            for j in range(s, e - 1)[::-1]:
                beta[j, :] = logsumexp((emissions[j + 1, :] +
                                        beta[j + 1, :]).reshape(1, -1) +
                                       transitions, axis=1)
        for view in views:
            view.close()
        return

    def expectation_step(self):
        futures = []
        for i in range(self.thread_seq_indices.shape[0] - 1):
            s, e = self.thread_seq_indices[i:i+2]
            futures.append(self.pool.apply_async(
                self._expectation_thread,
                args=(s, e, self.smm_map)))
        for f in futures:
            _ = f.get()
        return

    @classmethod
    def _expectation_thread(cls, start, end, smm_map):
        views = []
        views.append(SharedMemory(smm_map['sizes']))
        sizes = np.ndarray(6, np.int64, buffer=views[-1].buf)
        stateN, distN, obsN, seqN = sizes[:4]
        views.append(SharedMemory(smm_map['alpha']))
        alpha = np.ndarray((obsN, stateN), np.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['beta']))
        beta = np.ndarray((obsN, stateN), np.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['lngamma']))
        lngamma = np.ndarray((obsN, stateN), np.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['gamma']))
        gamma = np.ndarray((obsN, stateN), np.float64, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['obs_indices']))
        obs_indices = np.ndarray((seqN + 1), np.int64, buffer=views[-1].buf)

        s = obs_indices[start]
        e = obs_indices[end]
        lngamma[s:e, :] = alpha[s:e, :] + beta[s:e, :]
        for i in range(start, end):
            s, e = obs_indices[i:i+2]
            n = e - s
        s = obs_indices[start]
        e = obs_indices[end]
        nz = lngamma[s:e, :] > -np.inf
        gamma[s:e, :] = lngamma[s:e, :] - np.amax(lngamma[s:e, :],
                                                  axis=1, keepdims=True)
        gamma[s:e, :][np.logical_not(nz)] = 0
        gamma[s:e, :][nz] = np.exp(gamma[s:e, :][nz])
        gamma[s:e, :] /= np.sum(gamma[s:e, :], axis=1, keepdims=True)
        for view in views:
            view.close()
        return

    def maximization_step(self, **kwargs):
        futures = []
        for i in range(self.num_states):
            for j in range(self.num_distributions):
                futures.append(self.pool.apply_async(
                    self._maximize_emissions_thread,
                    args=(i, j, self.distributions[i][j], self.smm_map)))
        for f in futures:
            sIdx, dIdx, params = f.get()
            self.distributions[sIdx][dIdx].set_parameters(**params)

        futures = []
        for i in range(self.thread_seq_indices.shape[0] - 1):
            s, e = self.thread_seq_indices[i:i+2]
            futures.append(self.pool.apply_async(
                self._maximize_transitions_thread,
                args=(s, e, self.smm_map)))
        initial_probabilities = np.zeros_like(self.initial_probabilities)
        transition_matrix = np.zeros_like(self.transition_matrix)
        for f in futures:
            tmp = f.get()
            initial_probabilities += tmp[0]
            transition_matrix += tmp[1]
        initial_probabilities /= np.sum(initial_probabilities)
        transition_matrix /= np.sum(transition_matrix, axis=1, keepdims=True)
        self.initial_probabilities[:] = self.to_log(initial_probabilities)
        self.transition_matrix[:, :] = self.to_log(transition_matrix)
        return

    @classmethod
    def _maximize_emissions_thread(cls, sIdx, dIdx, distribution, smm_map):
        views = []
        views.append(SharedMemory(smm_map['sizes']))
        sizes = np.ndarray(6, np.int64, buffer=views[-1].buf)
        stateN, distN, obsN, seqN = sizes[:4]
        views.append(SharedMemory(smm_map['obs']))
        obs = np.ndarray((obsN, distN), np.int32, buffer=views[-1].buf)
        views.append(SharedMemory(smm_map['gamma']))
        gamma = np.ndarray((obsN, stateN), np.float64, buffer=views[-1].buf)
        params = distribution.optimize_parameters(obs[:, dIdx], gamma[:, sIdx])
        for view in views:
            view.close()
        return sIdx, dIdx, params

    @classmethod
    def _maximize_transitions_thread(cls, start, end, smm_map):
        views = []
        views.append(SharedMemory(smm_map['sizes']))
        sizes = np.ndarray(6, np.int64, buffer=views[-1].buf)
        stateN, distN, obsN, seqN = sizes[:4]
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

        for view in views:
            view.close()
        return np.exp(new_init_probs), np.exp(new_trans)

    def save(self, fname, **kwargs):
        data = {k: v for k, v in kwargs.items() if k != "distributions"}
        for name in ['transition_matrix', 'initial_probabilities']:
            if name not in data:
                data[name] = getattr(self, name)
        if 'distributions' in kwargs:
            distributions = kwargs['distributions']
        else:
            distributions = self.distributions
        distN = len(distributions[0])
        stateN = len(distributions)
        data['dist_types'] = np.zeros((distN), "<U5")
        for i in range(distN):
            data['dist_types'][i] = distributions[0][i].short_name
            for j in range(stateN):
                params = distributions[j][i].params
                dtype = [(k, type(v)) for k, v in params.items()]
                dname = f'dist.{j}.{i}'
                data[dname] = np.zeros(1, dtype=np.dtype(dtype))
                for k, v in params.items():
                    data[dname][k][0] = v
        np.savez(fname, **data)
        return

    def load(self, fname):
        temp = np.load(fname)
        for name in temp.keys():
            if not name.startswith('dist'):
                setattr(self, name, temp[name])
        dist_types = temp['dist_types']
        self.num_states = self.initial_probabilities.shape[0]
        self.num_distributions = dist_types.shape[0]
        self.distributions = []
        for i in range(self.num_states):
            self.distributions.append([])
            for j in range(self.num_distributions):
                self.distributions[i].append(Distribution(dist_types[j],
                                                          self.RNG.spawn(1)[0]))
                dist = temp[f"dist.{i}.{j}"]
                params = {}
                for name in dist.dtype.names:
                    params[name] = dist[name][0]
                self.distributions[i][j].set_parameters(**params)
        self.num_obs = 0
        self.num_seqs = 0
        self.num_maxes = 0
        return











































