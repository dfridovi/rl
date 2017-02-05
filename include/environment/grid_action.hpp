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
// Defines the GridAction enum.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef RL_ENVIRONMENT_GRID_ACTION_H
#define RL_ENVIRONMENT_GRID_ACTION_H

#include <random>

namespace rl {

  struct GridAction {
  public:
    enum Direction {UP, DOWN, LEFT, RIGHT};

    // Constructor/destructor. Default constructor generates a random action.
    // If a direction is provided, it is set.
    ~GridAction() {}
    GridAction()
      : direction_(static_cast<Direction>(unif_(rng_))) {}
    GridAction(Direction direction)
      : direction_(direction) {}

    // Define the boolean equality operator.
    bool operator==(const GridAction& rhs) const {
      return direction_ == rhs.direction_;
    }

    bool operator==(GridAction::Direction rhs) const {
      return direction_ == rhs;
    }

    // Public member variable.
    Direction direction_;

  private:
    // Static random number generator.
    static std::random_device rd_;
    static std::default_random_engine rng_;
    static std::uniform_int_distribution<size_t> unif_;

  }; //\struct GridAction

}  //\namespace rl

#endif
