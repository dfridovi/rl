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
// Defines the GridWorld class, which inherits from Environment.
//
///////////////////////////////////////////////////////////////////////////////

#include <environment/grid_world.hpp>
#include <util/types.h>

#include <glog/logging.h>
#include <limits>

namespace rl {

  GridWorld::~GridWorld() {}
  GridWorld::GridWorld(size_t nrows, size_t ncols, const GridState& goal)
    : DiscreteEnvironment<GridState, GridAction>(),
      nrows_(nrows), ncols_(ncols), goal_(goal) {}

  // Implement pure virtual method from Environment, but leave it virtual
  // so that a derived class can override it (for example, adding some
  // randomness, or other biases).
  double GridWorld::Simulate(GridState& state, const GridAction& action) const {
    // Handle all edge cases.
    if (state.ii_ == 0 && action == GridAction::Direction::UP)
      return kInvalidReward;
    else if (state.ii_ == nrows_ - 1 && action == GridAction::Direction::DOWN)
      return kInvalidReward;
    else if (state.jj_ == 0 && action == GridAction::Direction::LEFT)
      return kInvalidReward;
    else if (state.jj_ == ncols_ - 1 && action == GridAction::Direction::RIGHT)
      return kInvalidReward;

    // Definitely going to stay on the grid, so just parse action normally.
    switch (action.direction_) {
    case GridAction::Direction::UP :
      state.ii_--;
      break;
    case GridAction::Direction::DOWN :
      state.ii_++;
      break;
    case GridAction::Direction::LEFT :
      state.jj_--;
      break;
    case GridAction::Direction::RIGHT :
      state.jj_++;
      break;

    default :
      // Should never get here.
      LOG(ERROR) << "GridWorld: Error! Parsing unknown action.";
      return kInvalidReward;
    }

    // Return whether or not we are in the goal state. Penalize not being in
    // the goal state in order to get short paths.
    const double reward = (state == goal_) ? 1.0 : -1.0;
    return reward;
  }

  // Implement pure virtual methods to enumerate all states, and all actions
  // from a given state.
  void GridWorld::States(std::vector<GridState>& states) const {
    states.clear();

    for (size_t ii = 0; ii < nrows_; ii++)
      for (size_t jj = 0; jj < ncols_; jj++)
        states.push_back(GridState(ii, jj));
  }

  void GridWorld::Actions(const GridState& state,
                          std::vector<GridAction>& actions) const {
    actions.clear();

    // Catch invalid states.
    CHECK_LT(state.ii_, nrows_);
    CHECK_LT(state.jj_, ncols_);

    // Create a list of possible next states.
    const std::vector<GridAction::Direction> possibilities =
      {GridAction::Direction::UP, GridAction::Direction::DOWN,
       GridAction::Direction::LEFT, GridAction::Direction::RIGHT};

    for (const auto& direction : possibilities) {
      // Catch edge cases.
      if (state.ii_ == 0 && direction == GridAction::Direction::UP)
        continue;
      if (state.ii_ == nrows_ - 1 && direction == GridAction::Direction::DOWN)
        continue;
      if (state.jj_ == 0 && direction == GridAction::Direction::LEFT)
        continue;
      if (state.jj_ == ncols_ - 1 && direction == GridAction::Direction::RIGHT)
        continue;

      // If we get here, the direction must be valid.
      actions.push_back(GridAction(direction));
    }
  }

}  //\namespace rl
