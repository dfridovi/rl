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
// Defines a continuous epsilon-greedy policy.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef RL_POLICY_CONTINUOUS_EPSILON_GREEDY_POLICY_H
#define RL_POLICY_CONTINUOUS_EPSILON_GREEDY_POLICY_H

#include "../value/continuous_action_value_functor.hpp"
#include "../environment/continuous_environment.hpp"

#include <glog/logging.h>
#include <random>

namespace rl {

  template<typename StateType, typename ActionType>
  class ContinuousEpsilonGreedyPolicy {
  public:
    ~ContinuousEpsilonGreedyPolicy() {}
    explicit ContinuousEpsilonGreedyPolicy(double epsilon)
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

    // Return an action given the current state. If state is not valid,
    // returns false.
    bool Act(const ContinuousActionValue<StateType, ActionType>::ConstPtr& value,
             const ContinuousEnvironment<StateType, ActionType>& environment,
             const StateType& state, ActionType& action) const;

  private:
    // Epsilon value.
    double epsilon_;

    // Random number generator.
    static std::random_device rd_;
    static std::default_random_engine rng_;
  }; //\class ContinuousEpsilonGreedyPolicy

// ---------------------------- IMPLEMENTATION ------------------------------ //

  template<typename StateType, typename ActionType> std::random_device
  ContinuousEpsilonGreedyPolicy<StateType, ActionType>::rd_;

  template<typename StateType, typename ActionType> std::default_random_engine
  ContinuousEpsilonGreedyPolicy<StateType, ActionType>::rng_(rd_());

  // Act epsilon-greedily in the given state.
  template<typename StateType, typename ActionType>
  bool ContinuousEpsilonGreedyPolicy<StateType, ActionType>::Act(
     const ContinuousActionValue<StateType, ActionType>::ConstPtr& value,
     const ContinuousEnvironment<StateType, ActionType>& environment,
     const StateType& state, ActionType& action) const {
    CHECK_NOTNULL(value.get());

    // With probability epsilon, pick a random action.
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    if (unif(rng_) < epsilon_) {
      ActionType random_action;
      while (!environment.IsValid(state, random_action))
        random_action = ActionType();

      action = random_action;
    } else {
      if (!value->OptimalAction(state, action)) {
        LOG(WARNING) << "ContinuousEpsilonGreedyPolicy: Could not "
                     << "find optimal action.";
        return false;
      }
    }

    return true;
  }

}  //\namespace rl

#endif
