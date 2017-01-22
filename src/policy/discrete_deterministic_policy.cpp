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
// Defines the DiscreteDeterministicPolicy class, which derives from the
// Policy base class.
//
// Discrete deterministic policies are represented by an (unordered) map from
// states to actions.
//
///////////////////////////////////////////////////////////////////////////////

#include <policy/discrete_deterministic_policy.hpp>

namespace rl {

  // Construct from a state value function.
  template<typename StateType, typename ActionType>
  DiscreteDeterministicPolicy<StateType, ActionType>::
  DiscreteDeterministicPolicy(const DiscreteStateValueFunctor<StateType>& V) {
    // TODO!
  }

  // Construct from a state-action value function.
  template<typename StateType, typename ActionType>
  DiscreteDeterministicPolicy<StateType, ActionType>::
  DiscreteDeterministicPolicy(
    const DiscreteActionValueFunctor<StateType, ActionType>& Q) {
    // TODO!
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
