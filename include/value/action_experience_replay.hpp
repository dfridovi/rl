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
// Defines the ActionExperienceReplay class, which is used to store tuples of
// (state, action, value), which may be used to train a function approximator
// derived from the ContinuousActionValueFunctor base class.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef RL_VALUE_ACTION_EXPERIENCE_REPLAY_H
#define RL_VALUE_ACTION_EXPERIENCE_REPLAY_H

#include <vector>
#include <random>

namespace rl {

  template<typename StateType, typename ActionType>
  class ActionExperienceReplay {
  public:
    ~ActionExperienceReplay() {}
    explicit ActionExperienceReplay()
      : rd_(), rng_(rd_()) {}

    // Add experience to the dataset.
    void Add(const StateType& state, const ActionType& action, double value) {
      states_.push_back(state);
      actions_.push_back(action);
      values_.push_back(value);
    }

    // Sample a random element from the data. Returns true if successful.
    bool Sample(StateType& state, ActionType& action, double& value) {
      if (states_.size() == 0) {
        LOG(WARNING) << "ActionExperienceReplay: tried to sample "
                     << "from an empty dataset.";
        return false;
      }

      // Create a random int distribution.
      std::uniform_int_distribution<size_t> unif(0, states_.size() - 1);

      // Sample and return.
      const size_t ii = unif(rng_);
      state = states_[ii];
      action = actions_[ii];
      value = values_[ii];
      return true;
    }

  private:
    // Parallel lists of states, actions, and values.
    std::vector<StateType> states_;
    std::vector<ActionType> actions_;
    std::vector<double> values_;

    // Random number generation.
    std::random_device rd_;
    std::default_random_engine rng_;
  }; //\struct ActionValueFunctor

}  //\namespace rl

#endif
