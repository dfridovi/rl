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
// Defines the DiscreteSarsaLambda solver class. Current implementation assumes
// discrete state and action spaces, and deterministic environments.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef RL_SOLVER_DISCRETE_SARSA_LAMBDA_H
#define RL_SOLVER_DISCRETE_SARSA_LAMBDA_H

#include "../policy/discrete_epsilon_greedy_policy.hpp"
#include "../value/discrete_action_value_functor.hpp"
#include "../solver/solver_params.hpp"

#include <glog/logging.h>
#include <vector>

namespace rl {

  template<typename StateType, typename ActionType>
  class DiscreteSarsaLambda {
  public:
    ~DiscreteSarsaLambda() {}

    // Initialize to a random policy.
    explicit DiscreteSarsaLambda(const StateType& initial_state,
                                 const SolverParams& params)
      : initial_state_(initial_state),
        discount_factor_(params.discount_factor_),
        lambda_(params.lambda_),
        alpha_(params.alpha_),
        num_rollouts_(params.num_rollouts_),
        rollout_length_(params.rollout_length_),
        max_iterations_(params.max_iterations_),
        initial_epsilon_(params.initial_epsilon_),
        policy_(params.initial_epsilon_) {
      value_ = DiscreteActionValue<StateType, ActionType>::Create();
    }

    // Solve the MDP defined by the given environment. Returns whether or not
    // iteration reached convergence. If convergence is reached, the solver
    // sets 'epsilon' to zero for the optimal policy.
    bool Solve(const DiscreteEnvironment<StateType, ActionType>& environment);

    // Get a const reference to the policy and value function.
    const DiscreteEpsilonGreedyPolicy<StateType, ActionType>& Policy() const {
      return policy_;
    }

    const typename DiscreteActionValue<StateType, ActionType>::ConstPtr&
    Value() const {
      return value_;
    }

  private:
    // Update policy to be greedy with respect to the current state
    // value function. Returns the total number of changes made; when this
    // number is zero, we have reached convergence.
    size_t UpdatePolicy(
       const DiscreteEnvironment<StateType, ActionType>& environment);

    // Estimate the value function for the current policy using SARSA(lambda).
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
    typename DiscreteActionValue<StateType, ActionType>::Ptr value_;
  }; //\class DiscreteSarsaLambda

// ---------------------------- IMPLEMENTATION ------------------------------ //

  // Solve the MDP defined by the given environment. Returns whether or not
  // iterations reached convergence.
  template<typename StateType, typename ActionType>
  bool DiscreteSarsaLambda<StateType, ActionType>::Solve(
     const DiscreteEnvironment<StateType, ActionType>& environment) {
    // Initialize policy randomly.
    policy_.SetRandomly(environment);

    // Initialize value function to zero.
    value_->Initialize(environment);

    // Run until convergence.
    bool has_converged = false;
    for (size_t ii = 1; ii <= max_iterations_ && !has_converged; ii++) {
      // Update value functor.
      UpdateValueFunction(environment);

      // Update policy greedily.
      const size_t policy_changes = UpdatePolicy(environment);
      has_converged = (policy_changes == 0);

#if 0
      std::cout << "Iteration " << ii << ": made "
                << policy_changes << " changes." << std::endl;
#endif

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
  size_t DiscreteSarsaLambda<StateType, ActionType>::UpdatePolicy(
     const DiscreteEnvironment<StateType, ActionType>& environment) {
    return policy_.SetGreedily(value_);
  }

  // Estimate the value function for the current policy using SARSA(lambda).
  template<typename StateType, typename ActionType>
  void DiscreteSarsaLambda<StateType, ActionType>::UpdateValueFunction(
     const DiscreteEnvironment<StateType, ActionType>& environment) {
    // Run the specified number of rollouts.
    for (size_t ii = 0; ii < num_rollouts_; ii++) {
      // Initialize the current state to the initial state.
      StateType current_state = initial_state_;
      ActionType current_action;
      CHECK(policy_.Act(environment, current_state, current_action));

      // Start an eligibility trace, which we represent as a value function.
      // All states/actions in the trace will be tracked as normal. If a
      // state/action pair is not in the trace yet, that means it has not
      // been visited this rollout.
      typename DiscreteActionValue<StateType, ActionType>::Ptr trace =
        DiscreteActionValue<StateType, ActionType>::Create();

      // Simulate the rollout.
      for (int jj = 0;
           jj < rollout_length_ || (rollout_length_ < 0 &&
                                    !environment.IsTerminal(current_state));
           jj++) {

        // Increment elegibility trace at this state/action pair.
        const double current_trace = trace->Get(current_state, current_action);
        if (current_trace == kInvalidValue)
          trace->Set(current_state, current_action, 1.0);
        else
          trace->Set(current_state, current_action, current_trace + 1.0);

        // Get the current value at this state/action pair.
        const double current_value = value_->Get(current_state, current_action);

        // Simulate this action.
        const double reward =
          environment.Simulate(current_state, current_action);

        // Sample another action from the policy, but at the new state.
        CHECK(policy_.Act(environment, current_state, current_action));

        // Compute SARSA delta.
        const double next_value = value_->Get(current_state, current_action);
        const double delta =
           reward + discount_factor_ * next_value - current_value;

        // Update values and decay eligibility trace.
        for (auto& state_entry : trace->value_) {
          for (auto& action_entry : state_entry.second) {
            value_->Reference(state_entry.first, action_entry.first) +=
              alpha_ * delta * action_entry.second;
            action_entry.second *= discount_factor_ * lambda_;
          }
        }
      }
    }
  }
}  //\namespace rl

#endif
