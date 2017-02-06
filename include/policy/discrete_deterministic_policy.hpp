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
// Defines the DiscretePolicy class, which derives from the Policy base class.
//
// Discrete policies are represented by an (unordered) map from states to
// actions.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef RL_POLICY_DISCRETE_DETERMINISTIC_POLICY_H
#define RL_POLICY_DISCRETE_DETERMINISTIC_POLICY_H

#include <policy/policy.hpp>
#include <value/discrete_state_value_functor.hpp>
#include <value/discrete_action_value_functor.hpp>
#include <environment/discrete_environment.hpp>

#include <unordered_map>
#include <random>

namespace rl {

  template<typename StateType, typename ActionType>
  class DiscreteDeterministicPolicy : public Policy<StateType, ActionType> {
  public:
    ~DiscreteDeterministicPolicy() {}
    explicit DiscreteDeterministicPolicy() {}

    // Set to random valid actions in the given environment.
    void SetRandomly(
       const DiscreteEnvironment<StateType, ActionType>& environment);

    // Set to the greedy policy given a state value function V or an
    // action value function Q.
    void SetGreedily(
       const DiscreteStateValueFunctor<StateType>& V,
       const DiscreteEnvironment<StateType, ActionType>& environment);
    void SetGreedily(
       const DiscreteActionValueFunctor<StateType, ActionType>& Q);

    // Act deterministically at every state.
    bool Act(const StateType& state, ActionType& action) const;

  private:
    // Hash table to map states to actions.
    std::unordered_map<StateType, ActionType> policy_;
  }; //\class DiscreteDeterministicPolicy

// ---------------------------- IMPLEMENTATION ------------------------------ //

  // Set to random valid actions in the given environment.
  template<typename StateType, typename ActionType>
  void DiscreteDeterministicPolicy<StateType, ActionType>::SetRandomly(
     const DiscreteEnvironment<StateType, ActionType>& environment) {
    // Get a list of all the states in the environment.
    std::vector<StateType> states;
    environment.States(states);

    // Create a random number generator.
    std::random_device rd;
    std::default_random_engine rng(rd());

    // For each state, get a list of all the possible actions and select
    // one at random.
    std::vector<ActionType> actions;
    for (const auto& state : states) {
      environment.Actions(state, actions);

      // Choose a random action from the list.
      std::uniform_int_distribution<size_t> unif(0, actions.size() - 1);
      const ActionType random_action = actions[unif(rng)];

      // Check if 'policy_' does not yet contain this state.
      if (policy_.count(state) == 0)
        policy_.insert({state, random_action});
      else
        policy_.at(state) = random_action;
    }
  }

  // Set to the greedy policy given a state value function V.
  template<typename StateType, typename ActionType>
  void DiscreteDeterministicPolicy<StateType, ActionType>::SetGreedily(
     const DiscreteStateValueFunctor<StateType>& V,
     const DiscreteEnvironment<StateType, ActionType>& environment) {
    // Get a list of all the states in the environment.
    std::vector<StateType> states;
    environment.States(states);

    // Iterate over all states. For each one, find the action which
    // leads to the maximum value on the next step.
    std::vector<ActionType> actions;
    for (const auto& state : states) {
      environment.Actions(state, actions);

      // Simulate taking each action. Note that 'greedy' here means we
      // simply maximize V(next state), not including the actual
      // simulated reward.
      double max_value = -std::numeric_limits<double>::infinity();
      for (const auto& action : actions) {
        StateType next_state = state;
        const double reward = environment.Simulate(next_state, action);

        // Check if 'policy_' does not yet contain this state.
        if (policy_.count(state) == 0) {
          max_value = V(next_state);
          policy_.insert({state, action});
        } else {
          // Only update existing action if value has increased.
          if (V(next_state) > max_value) {
            max_value = V(next_state);
            policy_.at(state) = action;
          }
        }
      }
    }
  }

  // Set to the greedy policy given a state-action value function Q.
  template<typename StateType, typename ActionType>
  void DiscreteDeterministicPolicy<StateType, ActionType>::SetGreedily(
     const DiscreteActionValueFunctor<StateType, ActionType>& Q) {
    // Get a const refeence to the value table.
    const std::unordered_map<StateType, std::unordered_map<ActionType, double> >
      table = Q.ImmutableActionValueTable();

    // Iterate over all states.
    for (const auto& state_entry : table) {
      double max_value = -std::numeric_limits<double>::infinity();

      // Only update action if value is greater than previous max.
      for (const auto& action_entry : state_entry.second) {
        // Catch first time we see this state.
        if (policy_.count(state_entry.first) == 0) {
          policy_.insert({state_entry.first, action_entry.first});
          max_value = action_entry.second;
        } else {
          // Handle other actions.
          if (action_entry.second > max_value) {
            policy_.at(state_entry.first) = action_entry.first;
            max_value = action_entry.second;
          }
        }
      }
    }
  }

  // Act deterministically at every state.
  template<typename StateType, typename ActionType>
  bool DiscreteDeterministicPolicy<StateType, ActionType>::Act(
    const StateType& state, ActionType& action) const {
    // Check that 'state' is in the 'policy_'.
    if (policy_.count(state) == 0)
      return false;

    action = policy_.at(state);
    return true;
  }

}  //\namespace rl

#endif
