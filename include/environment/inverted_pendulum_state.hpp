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
// Defines the InvertedPendulumState struct.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef RL_ENVIRONMENT_INVERTED_PENDULUM_STATE_H
#define RL_ENVIRONMENT_INVERTED_PENDULUM_STATE_H

#include "../util/types.hpp"

#include <glog/logging.h>
#include <boost/functional/hash.hpp>
#include <stddef.h>
#include <math.h>
#include <random>

#ifdef SYSTEM_OSX
#include <GLUT/glut.h>
#endif

#ifdef SYSTEM_LINUX
#include <GL/glut.h>
#endif

namespace rl {

  struct InvertedPendulumState {
    // Angle from the positive x-axis, and its derivative.
    double theta_;
    double omega_;

    // Random number generator for initialization.
    static std::random_device rd_;
    static std::default_random_engine rng_;

    // Constructor/destructor.
    ~InvertedPendulumState() {}
    InvertedPendulumState(double theta, double omega)
      : theta_(theta), omega_(omega) {}
    InvertedPendulumState()
      : theta_(M_PI_2), omega_(0.0) {
      // Reset randomly.
      std::uniform_real_distribution<double> unif_theta(0.0, M_PI);
      std::uniform_real_distribution<double> unif_omega(-1.0, 1.0);

      theta_ = unif_theta(rng_);
      omega_ = unif_omega(rng_);
    }

    // Static number of dimensions.
    static constexpr size_t FeatureDimension() { return 2; }

    // Get a feature vector for this state.
    void Features(VectorXd& features) const {
      CHECK_EQ(features.size(), FeatureDimension());

      features(0) = theta_;
      features(1) = omega_;
    }

    // (In)equality operators. Note that these are not exactly transitive,
    // but rather are intended to simplify routine comparisons.
    bool operator==(const InvertedPendulumState& rhs) const {
      return (std::abs(theta_ - rhs.theta_) < 1e-8 &&
              std::abs(omega_ - rhs.omega_) < 1e-8);
    }

    bool operator!=(const InvertedPendulumState& rhs) const {
      return (std::abs(theta_ - rhs.theta_) >= 1e-8 ||
              std::abs(omega_ - rhs.omega_) >= 1e-8);
    }

    // OpenGL visualization. Must provide pendulum arm length and ball radius.
    void Visualize(double arm_length, double ball_radius) const {
      const size_t kNumVertices = 100;

      // Extract current position on xy plane.
      const GLfloat current_x = arm_length * std::cos(theta_);
      const GLfloat current_y = arm_length * std::sin(theta_);

      // Draw the arm first. Arm will be drawn as a narrow rectangle from the
      // origin to the center of the ball.
      const GLfloat arm_width = 0.25 * ball_radius;
      glBegin(GL_POLYGON);
      glColor4f(0.3, 0.3, 0.3, 0.5);
      // Bottom left, bottom right, top right, top left.
      glVertex2f(-0.5 * arm_width * std::sin(theta_),
                 0.5 * arm_width * std::cos(theta_));
      glVertex2f(0.5 * arm_width * std::sin(theta_),
                 -0.5 * arm_width * std::cos(theta_));
      glVertex2f(current_x + 0.5 * arm_width * std::sin(theta_),
                 current_y - 0.5 * arm_width * std::cos(theta_));
      glVertex2f(current_x - 0.5 * arm_width * std::sin(theta_),
                 current_y + 0.5 * arm_width * std::cos(theta_));
      glEnd();

      // Now draw the ball.
      glBegin(GL_POLYGON);
      glColor4f(0.0, 0.8, 0.2, 0.5);
      for (size_t ii = 0; ii < kNumVertices; ii++) {
        const GLfloat angle = 2.0 * M_PI *
          static_cast<GLfloat>(ii) / static_cast<GLfloat>(kNumVertices);
        glVertex2f(current_x + ball_radius * std::cos(angle),
                   current_y + ball_radius * std::sin(angle));
      }
      glEnd();
    }

    // Hash functor. Should not really ever need to hash a pendulum state,
    // but we provide this functor just in case.
    struct Hash {
      size_t operator()(const InvertedPendulumState& state) const {
        size_t seed = 0;
        boost::hash_combine(seed, boost::hash_value(state.theta_));
        boost::hash_combine(seed, boost::hash_value(state.omega_));

        return seed;
      }
    }; //\struct Hash
  }; //\struct InvertedPendulumState
}  //\namespace rl

#endif
