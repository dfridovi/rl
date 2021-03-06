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
// Defines the ContinuousQLearning solver class. Current implementation assumes
// continuous state and discrete action spaces.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef RL_SOLVER_CONTINUOUS_Q_LEARNING_H
#define RL_SOLVER_CONTINUOUS_Q_LEARNING_H

#include "../value/continuous_action_value_functor.hpp"
#include "../policy/continuous_epsilon_greedy_policy.hpp"
#include "../value/experience_replay.hpp"
#include "../solver/solver_params.hpp"

#include <glog/logging.h>
#include <vector>

namespace rl {

  template<typename StateType, typename ActionType>
  class ContinuousQLearning {
  public:
    ~ContinuousQLearning() {}

    // Parse parameters.
    explicit ContinuousQLearning(const StateType& initial_state,
                                 const SolverParams& params)
      : initial_state_(initial_state),
        initial_epsilon_(params.initial_epsilon_),
        discount_factor_(params.discount_factor_),
        alpha_(params.alpha_),
        learning_rate_(params.learning_rate_),
        learning_rate_decay_(params.learning_rate_decay_),
        num_rollouts_(params.num_rollouts_),
        rollout_length_(params.rollout_length_),
        num_exp_replays_(params.num_exp_replays_),
        batch_size_(params.batch_size_),
        policy_(params.initial_epsilon_) {}

    // Runs Q Learning algorithm for the specified number of iterations.
    // Solution is stored in the provided continuous value functor.
    void Solve(const ContinuousEnvironment<StateType, ActionType>& environment,
               typename ContinuousActionValue<StateType, ActionType>::Ptr& value,
               bool verbose = false, bool random_initial_state = true);

  private:
    // Estimate the value function for the current policy using Q Learning,
    // from a single rollout. Takes in both a floating value functor and a
    // const fixed one for training.
    void UpdateValueFunction(
       const ContinuousEnvironment<StateType, ActionType>& environment,
       typename ContinuousActionValue<StateType, ActionType>::Ptr& value,
       bool random_initial_state);

    // Member variables.
    const StateType initial_state_;
    const double discount_factor_;
    const double initial_epsilon_;
    const double alpha_;
    double learning_rate_;
    const double learning_rate_decay_;
    const size_t num_exp_replays_;
    const size_t batch_size_;
    const size_t num_rollouts_;
    const int rollout_length_;

    ExperienceReplay<StateType, ActionType> replay_;
    ContinuousEpsilonGreedyPolicy<StateType, ActionType> policy_;
  }; //\class ContinuousQLearning

// ---------------------------- IMPLEMENTATION ------------------------------ //

  // Runs Q Learning algorithm for the specified number of iterations.
  // Solution is stored in the provided continuous value functor.
  template<typename StateType, typename ActionType>
  void ContinuousQLearning<StateType, ActionType>::Solve(
     const ContinuousEnvironment<StateType, ActionType>& environment,
     typename ContinuousActionValue<StateType, ActionType>::Ptr& value,
     bool verbose, bool random_initial_state) {
    // Run for the specified number of iterations. Assume value functor
    // has already been initialized.
    for (size_t ii = 1; ii <= num_rollouts_; ii++) {
      if (verbose)
        std::cout << "Training rollout #" << ii << "..." << std::flush;

      // Update the given value functor.
      UpdateValueFunction(environment, value, random_initial_state);

      if (verbose)
        std::cout << "done." << std::endl;

      // Update epsilon.
      policy_.SetEpsilon(initial_epsilon_ / static_cast<double>(ii + 1));
    }
  }

  // Estimate the value function for the current policy using Q Learning,
  // from a single rollout.
  template<typename StateType, typename ActionType>
  void ContinuousQLearning<StateType, ActionType>::UpdateValueFunction(
     const ContinuousEnvironment<StateType, ActionType>& environment,
     typename ContinuousActionValue<StateType, ActionType>::Ptr& value,
     bool random_initial_state) {
    // Make a const copy of the value functor.
    const auto fixed_value = value->Copy();

    // Initialize the current state.
    StateType current_state;

    if (!random_initial_state)
      current_state = initial_state_;

    // Get the optimal action in this state.
    ActionType current_action;
    if (!fixed_value->OptimalAction(current_state, current_action))
      LOG(WARNING) << "ContinuousQLearning: Could not find optimal action.";

    // Simulate the rollout.
    for (int ii = 0; (ii < rollout_length_) ||
           (rollout_length_ < 0 && !environment.IsTerminal(current_state));
         ii++) {
      // Take the current action.
      StateType next_state = current_state;
      const double reward = environment.Simulate(next_state, current_action);

      // Get epsilon-optimal next action.
      ActionType next_action;
      if (!policy_.Act(value, environment, next_state, next_action))
        LOG(WARNING) << "ContinuousQLearning: Policy error.";

      // Store this experience in the replay unit.
      replay_.Add(current_state, current_action, reward, next_state);

      // Update current state!
      current_state = next_state;
    }

    // Replay experience and each time update the value function.
    for (size_t ii = 0; ii < num_exp_replays_; ii++) {
      std::vector<StateType> sample_states, sample_next_states;
      std::vector<ActionType> sample_actions;
      std::vector<double> sample_rewards;

      CHECK(replay_.Sample(batch_size_, sample_states, sample_actions,
                           sample_rewards, sample_next_states));

      // Compute the Q-Learning target at each sample.
      std::vector<double> targets;
      for (size_t jj = 0; jj < sample_states.size(); jj++) {
        // Get optimal next action.
        ActionType optimal_next_action;
        if (!policy_.Act(value, environment,
                         sample_next_states[jj], optimal_next_action)) {
          LOG(WARNING) << "ContinuousQLearning: Policy error.";
        }

        const double sample_value =
          value->Get(sample_states[jj], sample_actions[jj]);
        const double td_delta =
          sample_rewards[jj] - sample_value + discount_factor_ *
          fixed_value->Get(sample_next_states[jj], optimal_next_action);
        const double target =
          sample_value + alpha_ * std::min(std::max(td_delta, -10.0), 10.0);

        targets.push_back(target);
      }

      // Update.
      const double loss =
        value->Update(sample_states, sample_actions, targets, learning_rate_);

      if (ii % 1000 == 0) {
        std::printf("Loss on replay %zu was %f.\n", ii, loss);
      }
    }

    // Decay learning rate.
    learning_rate_ *= learning_rate_decay_;
  }
}  //\namespace rl

#endif
