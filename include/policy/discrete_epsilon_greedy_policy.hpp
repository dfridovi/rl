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
// Defines a discrete epsilon-greedy policy, which derives from the Policy base.
//
// Discrete epsilon-greedy policies are represented by an (unordered) map from
// states to actions, and they also keep track of epsilon.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef RL_POLICY_DISCRETE_EPSILON_GREEDY_POLICY_H
#define RL_POLICY_DISCRETE_EPSILON_GREEDY_POLICY_H

#include "../value/discrete_state_value_functor.hpp"
#include "../value/discrete_action_value_functor.hpp"
#include "../environment/discrete_environment.hpp"

#include <glog/logging.h>
#include <unordered_map>
#include <random>

namespace rl {

  template<typename StateType, typename ActionType>
  class DiscreteEpsilonGreedyPolicy {
  public:
    ~DiscreteEpsilonGreedyPolicy() {}
    explicit DiscreteEpsilonGreedyPolicy(double epsilon)
      : epsilon_(epsilon) {
      CHECK_GE(epsilon, 0.0);
      CHECK_LE(epsilon, 1.0);
    }

    // Get/set epsilon.
    void Epsilon() const { return epsilon_; }
    void SetEpsilon(double epsilon) {
      CHECK_GE(epsilon, 0.0);
      CHECK_LE(epsilon, 1.0);
      epsilon_ = epsilon;
    }

    // Set to random valid actions in the given environment.
    void SetRandomly(
       const DiscreteEnvironment<StateType, ActionType>& environment);

    // Set to the greedy policy given a state value function V or an
    // action value function Q. Returns the total number of changes made.
    size_t SetGreedily(
       const DiscreteStateValueFunctor<StateType>& V,
       const DiscreteEnvironment<StateType, ActionType>& environment,
       double discount_factor);
    size_t SetGreedily(
       const DiscreteActionValueFunctor<StateType, ActionType>& Q);

    // Return an action given the current state. If state is not valid,
    // returns false.
    bool Act(const DiscreteEnvironment<StateType, ActionType>& environment,
             const StateType& state, ActionType& action) const;

  private:
    // Hash table to map states to actions. All StateTypes must implement
    // their own 'Hash' functors.
    std::unordered_map<StateType, ActionType, typename StateType::Hash> policy_;

    // Epsilon value.
    double epsilon_;

    // Random number generator.
    static std::random_device rd_;
    static std::default_random_engine rng_;
  }; //\class DiscreteEpsilonGreedyPolicy

// ---------------------------- IMPLEMENTATION ------------------------------ //

  template<typename StateType, typename ActionType> std::random_device
  DiscreteEpsilonGreedyPolicy<StateType, ActionType>::rd_;

  template<typename StateType, typename ActionType> std::default_random_engine
  DiscreteEpsilonGreedyPolicy<StateType, ActionType>::rng_(rd_());


  // Set to random valid actions in the given environment.
  template<typename StateType, typename ActionType>
  void DiscreteEpsilonGreedyPolicy<StateType, ActionType>::SetRandomly(
     const DiscreteEnvironment<StateType, ActionType>& environment) {
    // Get a list of all the states in the environment.
    std::vector<StateType> states;
    environment.States(states);

    // For each state, get a list of all the possible actions and select
    // one at random.
    std::vector<ActionType> actions;
    for (const auto& state : states) {
      environment.Actions(state, actions);

      // Choose a random valid action from the list.
      std::uniform_int_distribution<size_t> unif(0, actions.size() - 1);
      ActionType random_action = actions[unif(rng_)];
      while (!environment.IsValid(state, random_action))
        random_action = actions[unif(rng_)];

      // Check if 'policy_' does not yet contain this state.
      if (policy_.count(state) == 0)
        policy_.insert({state, random_action});
      else
        policy_.at(state) = random_action;
    }
  }

  // Set to the greedy policy given a state value function V. Returns
  // the total number of changes made.
  template<typename StateType, typename ActionType>
  size_t DiscreteEpsilonGreedyPolicy<StateType, ActionType>::SetGreedily(
     const DiscreteStateValueFunctor<StateType>& V,
     const DiscreteEnvironment<StateType, ActionType>& environment,
     double discount_factor) {
    // Iterate over all states. For each one, find the action which
    // leads to the maximum value on the next step.
    size_t num_changes = 0;
    std::vector<ActionType> actions;
    for (const auto& entry : V.value_) {
      const StateType state = entry.first;
      const double value = entry.second;

      // Get all feasible actions.
      environment.Actions(state, actions);

      // Flag for whether or not we have seen this state before.
      const bool is_new_state = (policy_.count(state) == 0);

      // Find best action from this state.
      double max_value = -std::numeric_limits<double>::infinity();
      ActionType best_action;
      for (const auto& action : actions) {
        StateType next_state = state;
        const double reward = environment.Simulate(next_state, action);
        const double value = discount_factor * V(next_state) + reward;

        // Only update existing action if value has increased.
        if (value > max_value) {
          max_value = value;
          best_action = action;
        }
      }

      // Handle new state separately.
      if (is_new_state) {
        num_changes++;
        policy_.insert({state, best_action});
      } else if (policy_.at(state) != best_action) {
        num_changes++;
        policy_.at(state) = best_action;
      }
    }

    return num_changes;
  }

  // Set to the greedy policy given a state-action value function Q.
  // Returns the total number of changes made.
  template<typename StateType, typename ActionType>
  size_t DiscreteEpsilonGreedyPolicy<StateType, ActionType>::SetGreedily(
     const DiscreteActionValueFunctor<StateType, ActionType>& Q) {
    // Iterate over all states.
    size_t num_changes = 0;
    for (const auto& state_entry : Q.value_) {
      const StateType state = state_entry.first;

      // Check if this is a new state or not.
      const bool is_new_state = (policy_.count(state) == 0);

      // Find best action.
      double max_value = -std::numeric_limits<double>::infinity();
      ActionType best_action;
      for (const auto& action_entry : state_entry.second) {
        if (action_entry.second > max_value) {
          max_value = action_entry.second;
          best_action = action_entry.first;
        }
      }

      // Handle new state separately.
      if (is_new_state) {
        num_changes++;
        policy_.insert({state, best_action});
      } else if (policy_.at(state) != best_action) {
        num_changes++;
        policy_.at(state) = best_action;
      }
    }

    return num_changes;
  }

  // Act deterministically at every state.
  template<typename StateType, typename ActionType>
  bool DiscreteEpsilonGreedyPolicy<StateType, ActionType>::Act(
     const DiscreteEnvironment<StateType, ActionType>& environment,
     const StateType& state, ActionType& action) const {
    // Check that 'state' is in the 'policy_'.
    if (policy_.count(state) == 0)
      return false;

    action = policy_.at(state);

    // With probability epsilon, pick a random action.
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    if (unif(rng_) < epsilon_) {
      ActionType random_action;
      while (!environment.IsValid(state, random_action))
        random_action = ActionType();

      action = random_action;
    }

    return true;
  }

}  //\namespace rl

#endif
