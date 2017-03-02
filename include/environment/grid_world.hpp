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

#ifndef RL_ENVIRONMENT_GRID_WORLD_H
#define RL_ENVIRONMENT_GRID_WORLD_H

#include "../environment/discrete_environment.hpp"
#include "../environment/grid_state.hpp"
#include "../environment/grid_action.hpp"

#include <stddef.h>

#ifdef SYSTEM_OSX
#include <GLUT/glut.h>
#endif

#ifdef SYSTEM_LINUX
#include <GL/glut.h>
#endif

namespace rl {

  class GridWorld : public DiscreteEnvironment<GridState, GridAction> {
  public:
    virtual ~GridWorld();
    explicit GridWorld(size_t nrows, size_t ncols, const GridState& goal);

    // Implement pure virtual method from Environment, but leave it virtual
    // so that a derived class can override it (for example, adding some
    // randomness, or other biases).
    virtual double Simulate(GridState& state, const GridAction& action) const;

    // Implement pure virtual method to return whether or not an action is
    // valid in a given state.
    virtual bool IsValid(const GridState& state,
                         const GridAction& action) const;

    // Implement pure virtual method to return whether a state is terminal.
    virtual bool IsTerminal(const GridState& state) const;

    // Implement pure virtual methods to enumerate all states, and all actions
    // from a given state.
    virtual void States(std::vector<GridState>& states) const;
    virtual void Actions(const GridState& state,
                         std::vector<GridAction>& actions) const;

    // Visualize using OpenGL.
    virtual void Visualize() const;

  protected:
    // Dimensions.
    const size_t nrows_;
    const size_t ncols_;

    // Goal state.
    const GridState goal_;
  }; //\class GridWorld

}  //\namespace rl

#endif
