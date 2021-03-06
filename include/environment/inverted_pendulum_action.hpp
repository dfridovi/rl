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
// Defines the InvertedPendulumAction struct.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef RL_ENVIRONMENT_INVERTED_PENDULUM_ACTION_H
#define RL_ENVIRONMENT_INVERTED_PENDULUM_ACTION_H

#include <util/types.hpp>

#include <glog/logging.h>
#include <functional>
#include <random>
#include <stddef.h>
#include <math.h>

namespace rl {

  struct InvertedPendulumAction {
    // Just a single torque input.
    double torque_;

    // Static min and max torques.
    static double min_torque_;
    static double max_torque_;

    // Static number of discrete values this action can take.
    // This will only be used to compute approximately-optimal actions.
    static size_t num_values_;

    // Random number generation.
    static std::random_device rd_;
    static std::default_random_engine rng_;

    // Constructor/destructor.
    ~InvertedPendulumAction() {}
    InvertedPendulumAction(double torque)
      : torque_(torque) {}
    InvertedPendulumAction() {
      std::uniform_real_distribution<double> unif(min_torque_, max_torque_);
      torque_ = unif(rng_);
    }

    // Static setters.
    static void SetLimits(double min_torque, double max_torque) {
      min_torque_ = min_torque;
      max_torque_ = max_torque;
    }

    static void SetNumValues(size_t num_values) {
      CHECK_GE(num_values, 2);
      num_values_ = num_values;
    }

    // Get a list of 'num_values_' possible actions.
    static void DiscreteValues(std::vector<InvertedPendulumAction>& actions) {
      actions.clear();

      for (size_t ii = 0; ii < num_values_; ii++) {
        const double torque =
          (static_cast<double>(ii) / static_cast<double>(num_values_ - 1)) *
           (max_torque_ - min_torque_) + min_torque_;
        actions.push_back(InvertedPendulumAction(torque));
      }
    }

    // Static number of dimensions and min/max values.
    static constexpr size_t FeatureDimension() { return 1; }

    static double MaxAlongDimension(size_t ii) {
      CHECK_LE(ii, FeatureDimension() - 1);
      return max_torque_;
    }

    static double MinAlongDimension(size_t ii) {
      CHECK_LE(ii, FeatureDimension() - 1);
      return min_torque_;
    }

    // Get a feature vector for this action.
    void Features(VectorXd& features) const {
      CHECK_EQ(features.size(), FeatureDimension());

      features(0) = torque_;
    }

    // Set from features.
    void FromFeatures(const VectorXd& features) {
      CHECK_EQ(features.size(), FeatureDimension());

      // Threshold.
      if (isnan(features(0)))
          torque_ = 0.5 * (max_torque_ + min_torque_);
      else
        torque_ = std::min(std::max(features(0), min_torque_), max_torque_);
    }

    // (In)equality operators. Note that these are not exactly transitive,
    // but rather are intended to simplify routine comparisons.
    bool operator==(const InvertedPendulumAction& rhs) const {
      return std::abs(torque_ - rhs.torque_) < 1e-8;
    }

    bool operator!=(const InvertedPendulumAction& rhs) const {
      return std::abs(torque_ - rhs.torque_) >= 1e-8;
    }

    // Hash functor. Should not really ever need to hash a pendulum action,
    // but we provide this functor just in case.
    struct Hash {
      size_t operator()(const InvertedPendulumAction& action) const {
        return std::hash<double>{}(action.torque_);
      }
    }; //\struct Hash
  }; //\struct InvertedPendulumAction
}  //\namespace rl

#endif
