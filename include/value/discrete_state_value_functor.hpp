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
// Defines the DiscreteStateValueFunctor class, which derives from the
// StateValueFunctor base class.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef RL_VALUE_DISCRETE_STATE_VALUE_FUNCTOR_H
#define RL_VALUE_DISCRETE_STATE_VALUE_FUNCTOR_H

#include <value/state_value_functor.hpp>

#include <unordered_map>
#include <limits>
#include <vector>

namespace rl {

  template<typename StateType>
  struct DiscreteStateValue : public StateValue<StateType> {
    // Hash table to store the value function.
    std::unordered_map<StateType, double, typename StateType::Hash> value_;

    virtual ~DiscreteStateValue() {}

    // Typedefs.
    typedef std::shared_ptr<DiscreteStateValue> Ptr;
    typedef std::shared_ptr<const DiscreteStateValue> ConstPtr;

    // Factory method.
    static Ptr Create() {
      Ptr ptr(new DiscreteStateValue);
      return ptr;
    }

    // Must implement a deep copy.
    Ptr Copy() const {
      Ptr ptr(new DiscreteStateValue(*this));
      return ptr;
    }

    // Set the value at the current state. If the state is already in the table
    // reset its value to what is given here.
    void Set(const StateType& state, double value) {
      if (value_.count(state) > 0)
        value_.at(state) = value;
      else
        value_.insert({state, value});
    }

    // Pure virtual method to output the value at a state.
    double Get(const StateType& state) const {
      if (value_.count(state) == 0)
        return -std::numeric_limits<double>::infinity();

      return value_.at(state);
    }

    // Operator to get a reference to the value at a state.
    double& Reference(const StateType& state) {
      return value_.at(state);
    }

    // Initialize all the given states to zero.
    void Initialize(const std::vector<StateType>& states) {
      value_.clear();

      for (const auto& state : states)
        value_.insert({state, 0.0});
    }


  private:
    explicit DiscreteStateValue()
      : StateValue<StateType>() {}


  }; //\struct DiscreteStateValue

}  //\namespace rl

#endif
