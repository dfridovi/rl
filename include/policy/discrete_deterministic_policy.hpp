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

#include <unordered_map>

namespace rl {

  template<typename StateType, typename ActionType>
  class DiscreteDeterministicPolicy : public Policy<StateType, ActionType> {
  public:
    ~DiscreteDeterministicPolicy() {}

    // Construct from a Q (action) value function.
    explicit DiscreteDeterministicPolicy(
       const DiscreteActionValueFunctor<StateType, ActionType>& Q);

    // Act deterministically at every state.
    bool Act(const StateType& state, ActionType& action) const;

  private:
    // Hash table to map states to actions.
    std::unordered_map<StateType, ActionType> map_;
  }; //\class DiscreteDeterministicPolicy

// ------------------------------ IMPLEMENTATION ---------------------------- //

  // Construct from a state-action value function. Simply chooses the action
  // that maximizes the state-action value function, for each state.
  template<typename StateType, typename ActionType>
  DiscreteDeterministicPolicy<StateType, ActionType>::
  DiscreteDeterministicPolicy(
    const DiscreteActionValueFunctor<StateType, ActionType>& Q)
    : Policy<StateType, ActionType>() {
    // Get a const refeence to the value table.
    const std::unordered_map<StateType, std::unordered_map<ActionType, double> >
                             table = Q.ImmutableActionValueTable();

    // Iterate over all states.
    for (const auto& state_entry : table) {
      double max_value = -std::numeric_limits<double>::infinity();

      // Only update action if value is greater than previous max.
      for (const auto& action_entry : state_entry.second) {
        // Catch first action.
        if (map_.count(state_entry.first) == 0) {
          map_.insert({state_entry.first, action_entry.first});
          max_value = action_entry.second;
        } else {
          // Handle other actions.
          if (action_entry.second > max_value) {
            map_.at(state_entry.first) = action_entry.first;
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
    // Check that 'state' is in the 'map_'.
    if (map_.count(state) == 0)
      return false;

    action = map_.at(state);
    return true;
  }

}  //\namespace rl

#endif
