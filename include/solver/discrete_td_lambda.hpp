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
// Defines the DiscreteTdLambda solver class. Current implementation assumes
// discrete state and action spaces, and deterministic environments.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef RL_SOLVER_DISCRETE_TD_LAMBDA_H
#define RL_SOLVER_DISCRETE_TD_LAMBDA_H

#include <policy/discrete_epsilon_greedy_policy.hpp>
#include <value/discrete_state_value_functor.hpp>
#include <solver/td_lambda_params.hpp>

#include <glog/logging.h>
#include <vector>

namespace rl {

  template<typename StateType, typename ActionType>
  class DiscreteTdLambda {
  public:
    ~DiscreteTdLambda() {}

    // Initialize to a random policy.
    explicit DiscreteTdLambda(const StateType& initial_state,
                              const TdLambdaParams& params)
      : initial_state_(initial_state),
        discount_factor_(params.discount_factor_),
        lambda_(params.lambda_),
        alpha_(params.alpha_),
        num_rollouts_(params.num_rollouts_),
        rollout_length_(params.rollout_length_),
        max_iterations_(params.max_iterations_),
        initial_epsilon_(params.initial_epsilon_),
        policy_(params.initial_epsilon_) {}

    // Solve the MDP defined by the given environment. Returns whether or not
    // iteration reached convergence. If convergence is reached, the solver
    // sets 'epsilon' to zero for the optimal policy.
    bool Solve(const DiscreteEnvironment<StateType, ActionType>& environment);

    // Get a const reference to the policy and value function.
    const DiscreteEpsilonGreedyPolicy<StateType, ActionType>& Policy() const {
      return policy_;
    }

    const DiscreteStateValueFunctor<StateType>& Value() const {
      return value_;
    }

  private:
    // Update policy to be greedy with respect to the current state
    // value function. Returns the total number of changes made; when this
    // number is zero, we have reached convergence.
    size_t UpdatePolicy(
       const DiscreteEnvironment<StateType, ActionType>& environment);

    // Estimate the value function for the current policy using TD(lambda).
    void UpdateValueFunction(
       const DiscreteEnvironment<StateType, ActionType>& environment);

    // Member variables: as defined in constructor, plus policy and value fn.
    const StateType initial_state_;
    const double discount_factor_;
    const double lambda_;
    const double alpha_;
    const size_t max_iterations_;
    const size_t num_rollouts_;
    const int rollout_length_;
    double initial_epsilon_;
    DiscreteEpsilonGreedyPolicy<StateType, ActionType> policy_;
    DiscreteStateValueFunctor<StateType> value_;
  }; //\class DiscreteTdLambda

// ---------------------------- IMPLEMENTATION ------------------------------ //

  // Solve the MDP defined by the given environment. Returns whether or not
  // iterations reached convergence.
  template<typename StateType, typename ActionType>
  bool DiscreteTdLambda<StateType, ActionType>::Solve(
     const DiscreteEnvironment<StateType, ActionType>& environment) {
    // Initialize policy randomly.
    policy_.SetRandomly(environment);

    // Initialize value function to zero.
    std::vector<StateType> states;
    environment.States(states);
    value_.Initialize(states);

    // Run until convergence.
    bool has_converged = false;
    for (size_t ii = 1; ii <= max_iterations_ && !has_converged; ii++) {
      // Update value functor.
      UpdateValueFunction(environment);

      // Update policy greedily.
      const size_t policy_changes = UpdatePolicy(environment);
      has_converged = (policy_changes == 0);

      // Decay epsilon.
      policy_.SetEpsilon(initial_epsilon_ / static_cast<double>(ii));
    }

    // Set epsilon to zero if converged.
    if (has_converged)
      policy_.SetEpsilon(0.0);

    return has_converged;
  }

  // Update policy to be greedy with respect to the current state
  // value function. Returns the total number of changes made; when this
  // number is zero, we have reached convergence.
  template<typename StateType, typename ActionType>
  size_t DiscreteTdLambda<StateType, ActionType>::UpdatePolicy(
     const DiscreteEnvironment<StateType, ActionType>& environment) {
    return policy_.SetGreedily(value_, environment, discount_factor_);
  }

  // Estimate the value function for the current policy using TD(lambda).
  template<typename StateType, typename ActionType>
  void DiscreteTdLambda<StateType, ActionType>::UpdateValueFunction(
     const DiscreteEnvironment<StateType, ActionType>& environment) {
    // Run the specified number of rollouts.
    for (size_t ii = 0; ii < num_rollouts_; ii++) {
      // Initialize the current state to the initial state.
      StateType current_state = initial_state_;

      // Start an eligibility trace. All states in the trace will be tracked
      // as normal. If a state is not in the trace yet, that means it has not
      // been visited this rollout.
      std::unordered_map<StateType, double, typename StateType::Hash> trace;

      // Simulate the rollout.
      for (int jj = 0;
           jj < rollout_length_ || (rollout_length_ < 0 &&
                                    !environment.IsTerminal(current_state));
           jj++) {
        // Increment elegibility trace at this state.
        if (trace.count(current_state) > 0)
          trace.at(current_state) += 1.0;
        else
          trace.insert({current_state, 1.0});

        // Get the action dictated by 'policy_'.
        ActionType action;
        CHECK(policy_.Act(environment, current_state, action));

        // Simulate this action and compute TD delta.
        const double current_value = value_(current_state);
        const double reward = environment.Simulate(current_state, action);
        const double delta =
           reward + discount_factor_ * value_(current_state) - current_value;

        // Update values and decay eligibility trace.
        for (auto& entry : trace) {
          value_[entry.first] += alpha_ * delta * entry.second;
          entry.second *= discount_factor_ * lambda_;
        }
      }
    }
  }
}  //\namespace rl

#endif
