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
    // updates per iteration and the discount factor.
    explicit ModifiedPolicyIteration(size_t num_value_updates,
                                     double discount_factor)
      : num_value_updates_(num_value_updates),
        discount_factor_(discount_factor) {}

    // Solve the MDP defined by the given environment.
    void Solve(const DiscreteEnvironment<StateType, ActionType>& environment);

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
    // discount factor, policy, and state value function.
    const size_t num_value_updates_;
    const double discount_factor_;
    DiscreteDeterministicPolicy<StateType, ActionType> policy_;
    DiscreteStateValueFunctor<StateType> value_;

  }; //\class ModifiedPolicyIteration

// ---------------------------- IMPLEMENTATION ------------------------------ //

  // Solve the MDP defined by the given environment.
  template<typename StateType, typename ActionType>
  void ModifiedPolicyIteration<StateType, ActionType>::Solve(
     const DiscreteEnvironment<StateType, ActionType>& environment) {
    // Initialize policy randomly.
    policy_.SetRandomly(environment);

    // Initialize value function to zero.
    std::vector<StateType> states;
    environment.States(states);
    value_.Initialize(states);

    // Run until convergence.
    bool has_converged = false;
    while (!has_converged) {
      // Update value functor the specified number of times.
      for (size_t ii = 0; ii < num_value_updates_; ii++) {
        const size_t value_changes = UpdateValueFunction(environment);

        // Break if converged.
        if (value_changes == 0)
          break;
      }

      // Update policy greedily.
      const size_t policy_changes = UpdatePolicy(environment);
      has_converged = (policy_changes == 0);
    }
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
    for (const auto& entry : value_.value_) {
      StateType next_state = entry.first;
      const ActionType action = policy_(next_state);

      // Simulate this action.
      const double reward = environment.Simulate(next_state, action);

      // Update the value at this state.
      value_.value_.at(entry.first) =
        value_(next_state) * discount_factor_ + reward;
    }
  }

}  //\namespace rl

#endif
