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
// Defines the DiscreteActionValueFunctor class, which derives from the
// ActionValueFunctor base class.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef RL_VALUE_DISCRETE_ACTION_VALUE_FUNCTOR_H
#define RL_VALUE_DISCRETE_ACTION_VALUE_FUNCTOR_H

#include <value/action_value_functor.hpp>
#include <environment/discrete_environment.hpp>
#include <util/types.hpp>

#include <unordered_map>
#include <limits>
#include <iostream>

namespace rl {

  template<typename StateType, typename ActionType>
  struct DiscreteActionValue :
    public ActionValue<StateType, ActionType> {
    // Hash table to store the value function. Both StateType and ActionType
    // must implement their own 'Hash' functors.
    std::unordered_map<
      StateType,
      std::unordered_map<ActionType, double, typename ActionType::Hash>,
      typename StateType::Hash> value_;

    // Typedefs.
    typedef std::shared_ptr<DiscreteActionValue> Ptr;
    typedef std::shared_ptr<const DiscreteActionValue> ConstPtr;

    // Destructor.
    ~DiscreteActionValue() {}

    // Factory method.
    static Ptr Create();

    // Must implement a deep copy.
    Ptr Copy() const;

    // Pure virtual method to output the value at a state/action pair.
    double Get(const StateType& state, const ActionType& action) const;

    // Set the value of a given state, action pair. If pair is already in
    // the table, updates the value to what is given here.
    void Set(const StateType& state, const ActionType& action, double value);

    // Operator to get a reference to the value at a state.
    double& Reference(const StateType& state, const ActionType& action);

    // Initialize all feasible actions at each state to zero.
    void Initialize(
      const DiscreteEnvironment<StateType, ActionType>& environment);

  private:
    explicit DiscreteActionValue()
      : ActionValue<StateType, ActionType>() {}

  }; //\class DiscreteActionValue

// ----------------------------- IMPLEMENTATION ----------------------------- //

  // Factory method.
  template<typename StateType, typename ActionType>
  DiscreteActionValue<StateType, ActionType>::Ptr DiscreteActionValue::
  Create() {
    DiscreteActionValue<StateType, ActionType>::Ptr
      ptr(new DiscreteActionValue<StateType, ActionType>());
    return ptr;
  }

  // Must implement a deep copy.
  template<typename StateType, typename ActionType>
  DiscreteActionValue<StateType, ActionType>::Ptr DiscreteActionValue::
  Copy() const {
    DiscreteActionValue<StateType, ActionType>::Ptr
      ptr(new DiscreteActionValue<StateType, ActionType>(*this));
    return ptr;
  }

  template<typename StateType, typename ActionType>
  double DiscreteActionValue<StateType, ActionType>::
  Get(const StateType& state, const ActionType& action) {
    if (value_.count(state) == 0)
      return kInvalidValue;

    if (value_.at(state).count(action) == 0)
      return kInvalidValue;

    return value_.at(state).at(action);
  }

  template<typename StateType, typename ActionType>
  void DiscreteActionValue<StateType, ActionType>::
  Set(const StateType& state, const ActionType& action, double value) {
    if (value_.count(state) > 0) {
      if (value_.at(state).count(action) > 0)
        // Table contains this pair.
        value_.at(state).at(action) = value;
      else
        // Table contains state but not action.
        value_.at(state).insert({action, value});
    } else {
      // Table does not contain this state.
      value_.insert({state, std::unordered_map<
        ActionType, double, typename ActionType::Hash>({{action, value}})});
    }
  }

  template<typename StateType, typename ActionType>
  double& DiscreteActionValue<StateType, ActionType>::
  Reference(const StateType& state, const ActionType& action) {
    return value_.at(state).at(action);
  }

  template<typename StateType, typename ActionType>
  void DiscreteActionValue<StateType, ActionType>::
  Initialize(const DiscreteEnvironment<StateType, ActionType>& environment) {
    value_.clear();

    // Get a list of all states.
    std::vector<StateType> states;
    environment.States(states);

    for (const auto& state : states) {
      // Get a list of all feasible actions in this state.
      std::vector<ActionType> actions;
      environment.Actions(state, actions);

      for (const auto& action : actions)
        Set(state, action, 0.0);
    }
  }

}  //\namespace rl

#endif
