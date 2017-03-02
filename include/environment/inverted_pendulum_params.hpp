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
// Inverted pendulum parameters.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef RL_ENVIRONMENT_INVERTED_PENDULUM_PARAMS_H
#define RL_ENVIRONMENT_INVERTED_PENDULUM_PARAMS_H

#include <stddef.h>

namespace rl {

  struct InvertedPendulumParams {
    // Length of pendulum arm.
    double arm_length_ = 1.0;

    // Ball radius and mass.
    double ball_radius_ = 0.1;
    double ball_mass_ = 1.0;

    // Friction torque.
    double friction_ = 0.01;

    // Upper and lower bounds on applied torque.
    double torque_lower_ = -0.1;
    double torque_upper_ = 0.1;

    // Numerical integration time step.
    double time_step_ = 0.001;

    // Control period. Control is piecewise constant on this time interval.
    double control_period_ = 0.01;

    // Number of discrete values the torque can take in the specified interval.
    size_t num_action_values_ = 5;
  }; //\ struct InvertedPendulumParams
}  //\namespace rl

#endif
