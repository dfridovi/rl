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
// Defines the ModifiedPolicyIteration class. Current implementation assumes
// discrete state and action spaces, and deterministic environments.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef RL_SOLVER_MODIFIED_POLICY_ITERATION_H
#define RL_SOLVER_MODIFIED_POLICY_ITERATION_H

#include <policy/discrete_deterministic_policy.hpp>
#include <value/discrete_state_value_functor.hpp>

#include <vector>

namespace rl {

  template<typename StateType, typename ActionType>
  class ModifiedPolicyIteration {
  public:
    ~ModifiedPolicyIteration() {}

    // Initialize to a random policy. Pass in the number of value function
    // updates per iteration, max number of iterations, and the discount factor.
    explicit ModifiedPolicyIteration(size_t num_value_updates,
                                     size_t max_iterations,
                                     double discount_factor)
      : num_value_updates_(num_value_updates),
        max_iterations_(max_iterations),
        discount_factor_(discount_factor) {}

    // Solve the MDP defined by the given environment. Returns whether or not
    // iteration reached convergence.
    bool Solve(const DiscreteEnvironment<StateType, ActionType>& environment);

  private:
    // Update policy to be greedy with respect to the current state
    // value function. Returns the total number of changes made; when this
    // number is zero, we have reached convergence.
    size_t UpdatePolicy(
       const DiscreteEnvironment<StateType, ActionType>& environment);

    // Single round of optimization of the state value function, i.e.
    // set V(s) <== V(environment(Pi(s)). Returns the total number of changes
    // made; when this number is zero, we have reached convergence.
    size_t UpdateValueFunction(
       const DiscreteEnvironment<StateType, ActionType>& environment);

    // Member variables: number of value updates per policy update,
    // max iterations, discount factor, policy, and state value function.
    const size_t num_value_updates_;
    const size_t max_iterations_;
    const double discount_factor_;
    DiscreteDeterministicPolicy<StateType, ActionType> policy_;
    DiscreteStateValueFunctor<StateType> value_;

  }; //\class ModifiedPolicyIteration

// ---------------------------- IMPLEMENTATION ------------------------------ //

  // Solve the MDP defined by the given environment. Returns whether or not
  // iterations reached convergence.
  template<typename StateType, typename ActionType>
  bool ModifiedPolicyIteration<StateType, ActionType>::Solve(
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
      // Update value functor the specified number of times.
      for (size_t jj = 0; jj < num_value_updates_; jj++) {
        const size_t value_changes = UpdateValueFunction(environment);

        // Break if converged.
        if (value_changes == 0)
          break;
      }

      // Update policy greedily.
      const size_t policy_changes = UpdatePolicy(environment);
      has_converged = (policy_changes == 0);
    }

    return has_converged;
  }

  // Update policy to be greedy with respect to the current state
  // value function. Returns the total number of changes made; when this
  // number is zero, we have reached convergence.
  template<typename StateType, typename ActionType>
  size_t ModifiedPolicyIteration<StateType, ActionType>::UpdatePolicy(
     const DiscreteEnvironment<StateType, ActionType>& environment) {
    return policy_.SetGreedily(value_, environment, discount_factor_);
  }

  // Single round of optimization of the state value function, i.e.
  // set V(s) <== V(environment(Pi(s)). Returns the total number of changes
  //  made; when this number is zero, we have reached convergence.
  template<typename StateType, typename ActionType>
  size_t ModifiedPolicyIteration<StateType, ActionType>::UpdateValueFunction(
     const DiscreteEnvironment<StateType, ActionType>& environment) {
    // Iterate over all states.
    size_t num_changes = 0;
    for (const auto& entry : value_.value_) {
      StateType next_state = entry.first;
      ActionType action;
      policy_.Act(next_state, action);

      // Simulate this action.
      const double reward = environment.Simulate(next_state, action);
      const double next_value = value_(next_state) * discount_factor_ + reward;

      if (std::abs(entry.second - next_value) > 1e-8)
        num_changes++;

      // Update the value at this state.
      value_.value_.at(entry.first) = next_value;
    }

    return num_changes;
  }

}  //\namespace rl

#endif
