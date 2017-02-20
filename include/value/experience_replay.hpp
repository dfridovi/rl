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
// Defines the ExperienceReplay class, which is used to store tuples of
// (state, action, reward, state).
//
///////////////////////////////////////////////////////////////////////////////

#ifndef RL_VALUE_EXPERIENCE_REPLAY_H
#define RL_VALUE_EXPERIENCE_REPLAY_H

#include <vector>
#include <random>

namespace rl {

  template<typename StateType, typename ActionType>
  class ExperienceReplay {
  public:
    ~ExperienceReplay() {}
    explicit ExperienceReplay()
      : rd_(), rng_(rd_()) {}

    // Add experience to the dataset.
    void Add(const StateType& state, const ActionType& action, double reward,
             const StateType& next_state) {
      states_.push_back(state);
      actions_.push_back(action);
      rewards_.push_back(value);
      next_states_.push_back(next_state);
    }

    // Sample a random element from the data. Returns true if successful.
    bool Sample(StateType& state, ActionType& action, double& reward,
                StateType& next_state) {
      if (states_.size() == 0) {
        LOG(WARNING) << "ExperienceReplay: tried to sample "
                     << "from an empty dataset.";
        return false;
      }

      // Create a random int distribution.
      std::uniform_int_distribution<size_t> unif(0, states_.size() - 1);

      // Sample and return.
      const size_t ii = unif(rng_);
      state = states_[ii];
      action = actions_[ii];
      reward = rewards_[ii];
      next_state = next_states_[ii];
      return true;
    }

  private:
    // Parallel lists of states, actions, rewards, next states.
    std::vector<StateType> states_;
    std::vector<ActionType> actions_;
    std::vector<double> rewards_;
    std::vector<StateType> next_states_;

    // Random number generation.
    std::random_device rd_;
    std::default_random_engine rng_;
  }; //\struct ExperienceReplay

}  //\namespace rl

#endif
