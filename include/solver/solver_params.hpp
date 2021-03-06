/*
 * Copyright (c) 2017, The Regents of the University of California (Regents).
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *
 *    3. Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Please contact the author(s) of this library if you have any questions.
 * Authors: David Fridovich-Keil   ( dfk@eecs.berkeley.edu )
 */

///////////////////////////////////////////////////////////////////////////////
//
// General struct for storing solver parameters. Not all of these parameters
// will be used by any given solver, but for simplicity we place them all in
// one place since so many are common to all/most solvers.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef RL_SOLVER_SOLVER_PARAMS_H
#define RL_SOLVER_SOLVER_PARAMS_H

namespace rl {

  struct SolverParams {
    // Discount factor.
    double discount_factor_ = 0.9;

    // Lambda value (for weighting contributions from old states).
    // This parameter is not used in Q Learning.
    double lambda_ = 0.5;

    // Alpha value for interpolating between TD return and current value.
    double alpha_ = 0.5;

    // Maximum number of iterations of value function and policy updates.
    // This parameter is not used in Q Learning.
    size_t max_iterations_ = 100;

    // Number of rollouts used to estimate the value function.
    size_t num_rollouts_ = 1;

    // Maximum rollout length. If -1, rollout until state is terminal.
    int rollout_length_ = -1;

    // Initial epsilon-value for epsilon-greedy policy.
    double initial_epsilon_ = 0.05;

    // Number of experience replays -- i.e. number of SGD updates from
    // experience replays.
    size_t num_exp_replays_ = 20;

    // Batch size. This is the number of experiences replays per SGD update.
    size_t batch_size_ = 32;

    // Learning rate for SGD update, with decay rate.
    double learning_rate_ = 0.01;
    double learning_rate_decay_ = 0.9;
  }; //\ struct TdLambdaParams
}  //\namespace rl

#endif
