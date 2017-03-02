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
// Defines the InvertedPendulum class, which inherits ContinuousEnvironment.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef RL_ENVIRONMENT_INVERTED_PENDULUM_H
#define RL_ENVIRONMENT_INVERTED_PENDULUM_H

#include "../environment/continuous_environment.hpp"
#include "../environment/inverted_pendulum_params.hpp"
#include "../environment/inverted_pendulum_state.hpp"
#include "../environment/inverted_pendulum_action.hpp"

#include <stddef.h>

#ifdef SYSTEM_OSX
#include <GLUT/glut.h>
#endif

#ifdef SYSTEM_LINUX
#include <GL/glut.h>
#endif

namespace rl {

  class InvertedPendulum :
    public ContinuousEnvironment<InvertedPendulumState, InvertedPendulumAction> {
  public:
    ~InvertedPendulum() {}

    // If goal state is not specified, it is assumed to be perfectly still
    // and upright.
    explicit InvertedPendulum(const InvertedPendulumParams& params);
    explicit InvertedPendulum(const InvertedPendulumState& goal,
                              const InvertedPendulumParams& params);

    // Implement pure virtual method from Environment.
    double Simulate(InvertedPendulumState& state,
                    const InvertedPendulumAction& action) const;

    // Implement pure virtual method to return whether or not an action is
    // valid in a given state.
    bool IsValid(const InvertedPendulumState& state,
                 const InvertedPendulumAction& action) const;

    // Implement pure virtual method to return whether a state is terminal.
    bool IsTerminal(const InvertedPendulumState& state) const;

    // Visualize using OpenGL.
    void Visualize() const;

  private:
    // Dimensions and moment of inertia.
    const double arm_length_;
    const double ball_radius_;
    const double moment_;

    // Friction torque.
    const double friction_;

    // Torque limits.
    const double torque_lower_;
    const double torque_upper_;

    // Angle limits.
    const double theta_lower_;
    const double theta_upper_;

    // Numerical integration time step.
    const double time_step_;

    // Control period. Control is piecewise constant on this time interval.
    const double control_period_;

    // Goal state.
    const InvertedPendulumState goal_;
  }; //\class InvertedPendulum

}  //\namespace rl

#endif
