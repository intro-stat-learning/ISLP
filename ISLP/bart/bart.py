
#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#   Modified for ISLP by Jonathan Taylor 2021

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import invgamma

from sklearn.base import RegressorMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_random_state

from joblib import Parallel, delayed

from .tree import Tree
from .likelihood import marginal_loglikelihood
from .particle_tree import ParticleTree

class BART(BaseEnsemble, RegressorMixin):
    """
    Particle Gibbs BART sampling step

    Parameters
    ----------

    num_particles : int
        Number of particles for the conditional SMC sampler. Defaults to 10
    max_stages : int
        Maximum number of iterations of the conditional SMC sampler. Defaults to 100.

    Notes
    -----

    This sampler is inspired by the [Lakshminarayanan2015] Particle Gibbs sampler, but introduces
    several changes. The changes will be properly documented soon.

    References
    ----------

    .. [Lakshminarayanan2015] Lakshminarayanan, B. and Roy, D.M. and Teh, Y. W., (2015),
        Particle Gibbs for Bayesian Additive Regression Trees.
        ArviX, `link <https://arxiv.org/abs/1502.04622>`__
    """

    def __init__(self,
                 num_trees=200,
                 num_particles=10,
                 max_stages=5000,
                 split_prob=lambda depth: 0.95/(1+depth)**2,
                 min_depth=0,
                 std_scale=2,
                 split_prior=None,
                 ndraw=10,
                 burnin=100,
                 sigma_prior=(5, 0.9),
                 num_quantile=50,
                 random_state=None,
                 n_jobs=-1):

        self.num_particles = num_particles
        self.max_stages = max_stages
        self.num_trees = num_trees
        self.split_prob = split_prob
        self.min_depth = min_depth
        self.std_scale = std_scale
        self.split_prior = split_prior

        self.ndraw = ndraw
        self.burnin = burnin

        self.sigma_prior = sigma_prior
        self.num_quantile = num_quantile
        
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Chipman's default for prior
        self.mu_prior_var_ = (0.5 / (self.std_scale * np.sqrt(self.num_trees)))**2
        self.mu_prior_mean_ = 0



    def predict(self,
                X):

        check_is_fitted(self)

        nsample = len(self.trees_sample_)
        output = np.zeros(X.shape[0], float)

        for trees in self.trees_sample_:
            for tree in trees:
                tree_fit = np.array([tree.predict_out_of_sample(x) for x in X])
                output += tree_fit
        output = output / nsample
        return self._inverse(output)

    def staged_predict(self,
                       X,
                       start_idx=0):

        check_is_fitted(self)

        trees_sample_ = self.trees_sample_[start_idx:]
        nsample = len(trees_sample_)
        output = np.zeros((nsample, X.shape[0]), float)

        for nstep, trees in enumerate(trees_sample_):
            for tree in trees:
                tree_fit = np.array([tree.predict_out_of_sample(x) for x in X])
                output[nstep] += tree_fit
                
        output = np.cumsum(output, 0) / (np.arange(nsample) + 1)[:,None]
        return self._inverse(output)

    def fit(self,
            X,
            Y,
            sample_weight=None):

        X = np.asarray(X)
        Y = np.asarray(Y)

        random_state = check_random_state(self.random_state)
        n_jobs = self.n_jobs
        if self.n_jobs <= 0:
            n_jobs = 1

        random_idx = random_state.randint(0,2**30-1,size=(n_jobs,)) # 2**31-1 should be OK for int32

        parallel = Parallel(n_jobs=len(random_idx))

        qvals = np.linspace(0, 100, self.num_quantile+2)[1:-1]
        if self.num_quantile is not None:
            X_quantiles = np.percentile(X,
                                        qvals,
                                        axis=0)
        else:
            X_quantiles = None

        X_missing = np.any(np.isnan(X))

        # Chipman's defaults according to Lakshminarayanan
        # scale to range [-0.5,0.5] 
        _Y_min, _Y_max = np.nanmin(Y), np.nanmax(Y)
        _forward = lambda y: ((y - _Y_min) / (_Y_max - _Y_min) - 0.5)
        self._inverse = lambda out: (out + 0.5) * (_Y_max - _Y_min) + _Y_min
        Y_shift = _forward(Y)

        sigmasq = np.var(Y_shift)
        A, q = self.sigma_prior
        sigma_prior_B = invgamma(A, 0, 1).ppf(q) * sigmasq

        # args for each job
        args = (X,
                X_quantiles,
                X_missing,
                _forward,
                Y_shift,
                sigma_prior_B)

        self.trees_sample_ = []
        self.variable_inclusion_ = []
        self.depths_ = []
        self.num_leaves_ = []
        
        work = parallel(delayed(clone(self)._gibbs_sample)(*(args + (rs,)))
                        for rs in random_idx)

        for (atts,
             batch_trees,
             variable_inclusion) in work:
            self.trees_sample_.extend(batch_trees)
            self.variable_inclusion_.extend(variable_inclusion)

        for key, value in atts.items():
            setattr(self, key, value)

        for particles in self.trees_sample_:
            self.depths_.append([tree._max_depth for tree in particles])
            self.num_leaves_.append([len(tree.idx_leaf_nodes) for tree in particles])

        self.variable_inclusion_ = np.array(self.variable_inclusion_)
        self.depths_ = np.array(self.depths_)
        self.num_leaves_ = np.array(self.num_leaves_)
        
        return self

    # Private methods

    def _gibbs_sample(self,
                      X,
                      X_quantiles,
                      X_missing,
                      _forward,
                      Y_shift,
                      sigma_prior_B,
                      random_state):

        random_state = check_random_state(random_state)

        variable_inclusion = []
        depths = []
        num_leaves = []
        
        num_observations_ = X.shape[0]
        self.num_variates_ = X.shape[1]
        available_predictors = list(range(self.num_variates_))

        init_mean = Y_shift.mean()    
        sum_trees_output = np.full_like(Y_shift, init_mean)

        init_value_ = init_mean / self.num_trees
        self.init_idx_ = np.arange(num_observations_, dtype=int)
        self.particle_indices_ = list(range(1, self.num_particles))

        split_prior = self.split_prior or np.ones(X.shape[1])
        ssv = SampleSplittingVariable(split_prior, random_state)

        sigmasq = np.var(Y_shift)

        # instantiate the particles
        
        self.all_particles_ = []
        sum_trees_output = 0

        init_resid = Y_shift - init_mean * (self.num_trees - 1) / self.num_trees
        for i in range(self.num_trees):
            new_tree = Tree.init_tree(
                tree_id = i,
                leaf_node_value=init_value_,
                idx_data_points=self.init_idx_)

            log_weight = marginal_loglikelihood(init_resid,
                                                sigmasq,
                                                self.mu_prior_mean_,
                                                self.mu_prior_var_)[0]                             
                                                
            p = ParticleTree(new_tree,
                             init_resid,
                             log_weight,
                             self.split_prob,
                             self.min_depth,
                             X_missing,
                             ssv,
                             available_predictors,
                             self.num_trees,
                             sigmasq,
                             self.mu_prior_mean_,
                             self.mu_prior_var_,
                             random_state)

            self.all_particles_.append(p)
            sum_trees_output += p.tree.predict_output()
            
        counter = 0
        batch_trees = []

        while True:
            (particle_trees,
             sum_trees_output,
             stats) = self._gibbs_step_trees(X,
                                             X_quantiles,
                                             Y_shift,
                                             sigmasq,
                                             sum_trees_output,
                                             random_state)
            sigmasq = self._gibbs_step_sigma(Y_shift - sum_trees_output,
                                             sigma_prior_B,
                                             random_state)
            if counter >= self.burnin:
                batch_trees.append(particle_trees)
                variable_inclusion.append(stats['variable_inclusion'])

            if len(batch_trees) == self.ndraw:
                break

            counter += 1

        variable_inclusion = np.array(variable_inclusion)

        atts = {'sigma_prior_B_': sigma_prior_B}

        return (atts,
                batch_trees,
                variable_inclusion)
                

    def _gibbs_step_sigma(self,
                          resid,
                          sigma_prior_B,
                          random_state):

        n = resid.shape[0]
        A = self.sigma_prior[0] + n / 2
        B = sigma_prior_B + (resid**2).sum() / 2
        
        return invgamma(A,
                        0,
                        B).rvs(random_state=random_state)
    
    def _gibbs_step_trees(self,
                          X,
                          X_quantiles,
                          Y,
                          sigmasq,
                          sum_trees_output,
                          random_state):

        variable_inclusion = np.zeros(self.num_variates_, int)

        total_stages = 0
            
        for tree_id in range(self.num_trees):
            # Generate an initial set of SMC particles
            # at the end of the algorithm we return one of these particles as the new tree

            # Compute the sum of trees without the tree we are attempting to replace

            cur_particle = self.all_particles_[tree_id]

            sum_trees_output_noi = sum_trees_output - cur_particle.tree.predict_output()
            resid_noi = Y - sum_trees_output_noi

            # set the resid and sigmasq of `cur_particle` to be current
            # and instantiate particles: single leaf trees with the same resid
            particles = self.init_particles(cur_particle,
                                            sigmasq,
                                            resid_noi)

            for t in range(self.max_stages):
                # sample each particle (try to grow each tree)

                for p in particles[1:]:
                    # this is log_likelihood_ratio for the split if there was one
                    # so if tree does not grow this is just 0
                    # line 9 of Algorithm 2 of Lakshminarayanan
                    tree_grew, left_node, right_node = p.sample_tree_sequential(
                        X,
                        X_quantiles,
                        resid_noi,
                    )
                    # line 12 of Algorithm 2 of Lakshminarayanan
                    if tree_grew:
                        p.log_weight += p.increment_loglikelihood(left_node,
                                                                  right_node)
                        
                # line 13 of Algorithm 2 of Lakshminarayanan
                W_t, normalized_weights = _normalize(particles)

                # line 14-15 of Algorithm 2 of Lakshminarayanan
                # Resample all but first particle
                re_n_w = normalized_weights[1:] / normalized_weights[1:].sum()
                new_indices = random_state.choice(self.particle_indices_,
                                                  size=len(self.particle_indices_),
                                                  p=re_n_w)
                particles[1:] = particles[new_indices]

                # Set the new weights
                for p in particles:
                    p.log_weight = W_t

                # Check if particles can keep growing, otherwise stop iterating
                non_available_nodes_for_expansion = []
                for p in particles[1:]:
                    if p.expansion_nodes:
                        non_available_nodes_for_expansion.append(0)
                if all(non_available_nodes_for_expansion):
                    break
            total_stages += t
            
            # Get the new tree and update
            new_particle = random_state.choice(particles, p=normalized_weights)
            new_particle.sample_values(resid_noi)
            new_tree = new_particle.tree
            new_particle.log_weight = W_t - np.log(len(particles))
            # now sample the mean parameters within each leaf

            self.all_particles_[new_tree.tree_id] = new_particle
            sum_trees_output = sum_trees_output_noi + new_tree.predict_output()

            for index in new_particle.used_variates:
                variable_inclusion[index] += 1

        stats = {"variable_inclusion": variable_inclusion}
        return [p.tree for p in self.all_particles_], sum_trees_output, stats

    def init_particles(self,
                       base_particle: ParticleTree,
                       sigmasq: float,
                       resid: np.ndarray) -> np.ndarray:
        """
        Initialize particles
        """
        p = base_particle

        # update the residual and compute the marginal likelihood of the tree
        p.set_resid(resid)
        p.sigmasq = sigmasq
        p.log_weight = p.marginal_loglikelihood()  
        particles = [p]

        resid_sum = resid.sum()
        root_weight = marginal_loglikelihood(resid,
                                             sigmasq,
                                             self.mu_prior_mean_,
                                             self.mu_prior_var_)[0]                             

        for _ in self.particle_indices_:
            new_tree = Tree.init_tree(
                tree_id = p.tree.tree_id,
                leaf_node_value=resid_sum / (resid.shape[0] * self.num_trees),
                idx_data_points=self.init_idx_)

            new_root = new_tree.get_node(0)
            new_root._response_moments = (resid.shape[0], resid_sum, None)

            new_particle = ParticleTree(new_tree,
                                        resid,
                                        root_weight,
                                        p.split_prob,
                                        p.min_depth,
                                        p.X_missing,
                                        p.ssv,
                                        p.available_predictors,
                                        p.m,
                                        p.sigmasq,
                                        p.mu_prior_mean,
                                        p.mu_prior_var,
                                        p.random_state)

            particles.append(new_particle)

        return np.array(particles)


# Private functions

def _normalize(particles: List[ParticleTree]) -> Tuple[float, np.ndarray]:
    """
    Use logsumexp trick to get W_t and softmax to get normalized_weights
    """
    log_w = np.array([p.log_weight for p in particles])
    log_w_max = log_w.max()
    log_w_ = log_w - log_w_max
    w_ = np.exp(log_w_)
    w_sum = w_.sum()
    W_t = log_w_max + np.log(w_sum) - np.log(log_w.shape[0])
    normalized_weights = w_ / w_sum
    # stabilize weights to avoid assigning exactly zero probability to a particle
    normalized_weights += 1e-12
    return W_t, normalized_weights

class SampleSplittingVariable(object):

    def __init__(self,
                 alpha_prior,
                 random_state):
        """
        Sample splitting variables proportional to `alpha_prior`.

        This is equivalent as sampling weights from a Dirichlet distribution with `alpha_prior`
        parameter and then using those weights to sample from the available spliting variables.
        This enforce sparsity.
        """
        self.enu = list(enumerate(np.cumsum(alpha_prior / alpha_prior.sum())))
        self.random_state = random_state
        
    def rvs(self):
        r = self.random_state.random()
        for i, v in self.enu:
            if r <= v:
                return i

