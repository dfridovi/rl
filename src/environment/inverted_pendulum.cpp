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

#include <environment/inverted_pendulum.hpp>
#include <util/types.hpp>

#include <math.h>

namespace rl {

  // If goal state is not specified, it is assumed to be perfectly still
  // and upright.
  InvertedPendulum::InvertedPendulum(const InvertedPendulumParams& params)
    : arm_length_(params.arm_length_),
      ball_radius_(params.ball_radius_),
      moment_(params.ball_mass_ * arm_length_ * arm_length_),
      friction_(params.friction_),
      torque_lower_(params.torque_lower_),
      torque_upper_(params.torque_upper_),
      theta_lower_(std::asin(ball_radius_ / arm_length_)),
      theta_upper_(M_PI - theta_lower_),
      time_step_(params.time_step_),
      control_period_(params.control_period_),
      goal_(InvertedPendulumState(M_PI_2, 0.0)) {
    InvertedPendulumAction::SetLimits(torque_lower_, torque_upper_);
  }

  InvertedPendulum::InvertedPendulum(const InvertedPendulumState& goal,
                                     const InvertedPendulumParams& params)
    : arm_length_(params.arm_length_),
      ball_radius_(params.ball_radius_),
      moment_(params.ball_mass_ * arm_length_ * arm_length_),
      friction_(params.friction_),
      torque_lower_(params.torque_lower_),
      torque_upper_(params.torque_upper_),
      theta_lower_(std::asin(ball_radius_ / arm_length_)),
      theta_upper_(M_PI - theta_lower_),
      time_step_(params.time_step_),
      control_period_(params.control_period_),
      goal_(goal) {
    InvertedPendulumAction::SetLimits(torque_lower_, torque_upper_);
  }

  // Implement pure virtual method from Environment. Compute net torque at the
  // joint and translate to an angular acceleration. Integrate this numerically
  // and return accumulated reward, which is a combination of the absolute
  // angular distance from the goal and the angular velocity (sign depending
  // on whether it is pointing the right direction or not).
  double InvertedPendulum::Simulate(InvertedPendulumState& state,
                                    const InvertedPendulumAction& action) const {
    // Compute net torque and angular acceleration.
    const double gravity = -9.81 * std::cos(state.theta_) * arm_length_;
    const double friction = (gravity < 0.0) ?
      std::min(-gravity, friction_) : std::max(-gravity, -friction_);
    const double torque = gravity + friction + action.torque_;
    const double acceleration = torque * moment_;

    // Numerical integration.
    double reward = 0.0;
    bool bounds_check = true;
    for (double t = 0.0; t <= control_period_; t += time_step_) {
      state.omega_ += acceleration * time_step_;
      state.theta_ += state.omega_ * time_step_;

      // Check if angle is out of bounds.
      bounds_check &=
        (state.theta_ >= theta_lower_ && state.theta_ <= theta_upper_);

      // Accumulate reward.
      if (state.theta_ > goal_.theta_)
        reward -= state.omega_;
      else
        reward += state.omega_;

      reward -= std::abs(state.theta_ - goal_.theta_);
    }

    // If ever went out of bounds, set to invalid reward.
    if (!bounds_check)
      reward = kInvalidReward;
    return reward;
  }

  // Implement pure virtual method to return whether or not an action is
  // valid in a given state. Simple bounds checking on action (torque).
  bool InvertedPendulum::IsValid(const InvertedPendulumState& state,
                                 const InvertedPendulumAction& action) const {
    return (action.torque_ >= torque_lower_ && action.torque_ <= torque_upper_);
  }

  // Implement pure virtual method to return whether a state is terminal.
  // Terminal states are those where the ball will have collided with the ground
  // or where the ball has reached the goal state.
  bool InvertedPendulum::IsTerminal(const InvertedPendulumState& state) const {
    if (state.theta_ < theta_lower_ || state.theta_ > theta_upper_)
      return true;

    if (state == goal_)
      return true;

    return false;
  }

  // Visualize using OpenGL.
  void InvertedPendulum::Visualize() const {
    glClear(GL_COLOR_BUFFER_BIT);
    const size_t kNumVertices = 100;

    // Draw the ground first.
    const GLfloat kGroundExtent = 10.0;
    glBegin(GL_POLYGON);
    glColor4f(0.3, 0.3, 0.3, 0.8);
    // Top left, bottom left, bottom right, top right.
    glVertex2f(-kGroundExtent, 0.0);
    glVertex2f(-kGroundExtent, -kGroundExtent);
    glVertex2f(kGroundExtent, -kGroundExtent);
    glVertex2f(kGroundExtent, 0.0);
    glEnd();

    // Draw a semi-circular pin joint at the origin.
    glBegin(GL_POLYGON);
    glColor4f(0.0, 0.2, 0.8, 0.5);
    for (size_t ii = 0; ii < kNumVertices; ii++) {
      const GLfloat angle = M_PI *
        static_cast<GLfloat>(ii) / static_cast<GLfloat>(kNumVertices - 1);
      glVertex2f(ball_radius_ * std::cos(angle),
                 ball_radius_ * std::sin(angle));
    }
    glEnd();
  }

}  //\namespace rl
