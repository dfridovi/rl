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
// Defines the DiscreteEnvironment base class, which derives from Environment.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef RL_ENVIRONMENT_DISCRETE_ENVIRONMENT_H
#define RL_ENVIRONMENT_DISCRETE_ENVIRONMENT_H

#include "../environment/environment.hpp"

#include <vector>

namespace rl {

  template<typename StateType, typename ActionType>
  class DiscreteEnvironment : public Environment<StateType, ActionType> {
  public:
    virtual ~DiscreteEnvironment() {}

    // Pure virtual method to output the next state, given that the actor
    // takes the specified action from the given state. Returns the reward.
    virtual double Simulate(StateType& state,
                            const ActionType& action) const = 0;

    // Pure virtual method to return whether or not an action is valid in a
    // given state.
    virtual bool IsValid(const StateType& state,
                         const ActionType& action) const = 0;

    // Pure virtual method to return whether a state is terminal.
    virtual bool IsTerminal(const StateType& state) const = 0;

    // Pure virtual methods to enumerate all states, and all actions from
    // a given state.
    virtual void States(std::vector<StateType>& states) const = 0;
    virtual void Actions(const StateType& state,
                         std::vector<ActionType>& actions) const = 0;

    // Pure virtual method to visualize (e.g. wuth OpenGL).
    virtual void Visualize() const = 0;

  protected:
    explicit DiscreteEnvironment()
      : Environment<StateType, ActionType>() {}
  }; //\class DiscreteEnvironment

}  //\namespace rl

#endif
