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

#include <glog/logging.h>

namespace rl {

  GridWorld::~GridWorld() {}
  GridWorld::GridWorld(size_t nrows, size_t ncols)
    : nrows_(nrows), ncols_(ncols) {}

  // Implement pure virtual method from Environment, but leave it virtual
  // so that a derived class can override it (for example, adding some
  // randomness, or other biases).
  bool GridWorld::Simulate(GridState& state, const GridAction& action) const {
    // Handle all edge cases.
    if (state.ii_ == 0 && action == GridAction::UP)
      return false;
    else if (state.ii_ == nrows_ - 1 && action == GridAction::DOWN)
      return false;
    else if (state.jj_ == 0 && action == GridAction::LEFT)
      return false;
    else if (state.jj_ == ncols_ - 1 && action == GridAction::RIGHT)
      return false;

    // Definitely going to stay on the grid, so just parse action normally.
    switch (action) {
    case GridAction::UP :
      state.ii_--;
      break;
    case GridAction::DOWN :
      state.ii_++;
      break;
    case GridAction::LEFT :
      state.jj_--;
      break;
    case GridAction::RIGHT :
      state.jj_++;
      break;

    default :
      // Should never get here.
      LOG(ERROR) << "GridWorld: Error! Parsing unknown action.";
      return false;
    }

    return true;
  }

}  //\namespace rl
