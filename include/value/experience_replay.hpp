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
#include <algorithm>
#include <unordered_set>

namespace rl {

  template<typename StateType, typename ActionType>
  class ExperienceReplay {
  public:
    ~ExperienceReplay() {}
    explicit ExperienceReplay()
      : rd_(), rng_(rd_()) {}

    // Add experience to the dataset.
    void Add(const StateType& state, const ActionType& action, double reward,
             const StateType& next_state);

    // Sample random elements from the data. Returns true if successful.
    bool Sample(size_t batch_size, std::vector<StateType>& states,
                std::vector<ActionType>& actions, std::vector<double>& rewards,
                std::vector<StateType>& next_states);

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

// ----------------------------- IMPLEMENTATION ----------------------------- //

  // Add experience to the dataset.
  template<typename StateType, typename ActionType>
  void ExperienceReplay<StateType, ActionType>::
  Add(const StateType& state, const ActionType& action, double reward,
      const StateType& next_state) {
    states_.push_back(state);
    actions_.push_back(action);
    rewards_.push_back(reward);
    next_states_.push_back(next_state);
  }

  // Sample random elements from the data. Returns true if successful.
  template<typename StateType, typename ActionType>
  bool ExperienceReplay<StateType, ActionType>::
  Sample(size_t batch_size, std::vector<StateType>& states,
         std::vector<ActionType>& actions, std::vector<double>& rewards,
         std::vector<StateType>& next_states) {
      states.clear();
      actions.clear();
      rewards.clear();
      next_states.clear();

      // Threshold batch size.
      size_t thresholded_batch_size = batch_size;
      if (batch_size > states_.size()) {
        LOG(WARNING) << "ExperienceReplay: Batch size is too large.";
        thresholded_batch_size = states_.size();
      }

      // Generate a random subset of the training data one of two ways:
      // (1) if batch size / total size >= 1 - 1/e, randomly shuffle,
      // (2) otherwise, draw random indices and check if they've been drawn.
      if (static_cast<double>(thresholded_batch_size) / states_.size() >=
          1.0 - 1.0 / M_E) {
        // (1) If batch size is large, do a random shuffle.
        std::vector<size_t> indices(states_.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng_);

        for (size_t ii = 0; ii < thresholded_batch_size; ii++) {
          states.push_back(states_[ indices[ii] ]);
          actions.push_back(actions_[ indices[ii] ]);
          rewards.push_back(rewards_[ indices[ii] ]);
          next_states.push_back(next_states_[ indices[ii] ]);
        }
      } else {
        // (2) Batch size is small, so choose random indices.
        std::uniform_int_distribution<size_t> unif(0, states_.size() - 1);
        std::unordered_set<size_t> sampled_indices;

        while (states.size() < thresholded_batch_size) {
          // Pick a random index in the training set that we have not seen yet.
          const size_t ii = unif(rng_);
          if (sampled_indices.count(ii) > 0)
            continue;

          sampled_indices.insert(ii);

          // Insert the corresponding samples.
          states.push_back(states_[ii]);
          actions.push_back(actions_[ii]);
          rewards.push_back(rewards_[ii]);
          next_states.push_back(next_states_[ii]);
        }
      }

      return states_.size() < batch_size;
    }

}  //\namespace rl

#endif
