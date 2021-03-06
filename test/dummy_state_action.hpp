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
////////////////////////////////////////////////////////////////////////////////
//
// Dummy state and action types for unit testing.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef RL_TEST_DUMMY_STATE_ACTION_H
#define RL_TEST_DUMMY_STATE_ACTION_H

#include <util/types.hpp>

#include <glog/logging.h>
#include <vector>
#include <random>

namespace rl {
  namespace test {
    // Dummy scalar state and action structs.
    struct DummyState {
      double state_;

      // Static rng.
      static std::random_device rd_;
      static std::default_random_engine rng_;
      static std::uniform_real_distribution<double> unif_;

      DummyState()
        : state_(unif_(rng_)) {}

      static constexpr size_t FeatureDimension() { return 1; }
      void Features(VectorXd& features) const {
        CHECK_EQ(features.size(), FeatureDimension());
        features(0) = state_;
      }
    }; //\ struct DummyState

    struct DummyAction {
      double action_;

      // Static rng.
      static std::random_device rd_;
      static std::default_random_engine rng_;
      static std::uniform_real_distribution<double> unif_;

      DummyAction()
        : action_(unif_(rng_)) {}

      static constexpr size_t FeatureDimension() { return 1; }
      void Features(VectorXd& features) const {
        CHECK_EQ(features.size(), FeatureDimension());
        features(0) = action_;
      }
      void FromFeatures(const VectorXd& features) {
        CHECK_EQ(features.size(), FeatureDimension());
        action_ = features(0);
      }

      static double MaxAlongDimension(size_t ii) {
        CHECK_LE(ii, FeatureDimension() - 1);
        return 1.0;
      }

      static double MinAlongDimension(size_t ii) {
        CHECK_LE(ii, FeatureDimension() - 1);
        return -1.0;
      }

      static void DiscreteValues(std::vector<DummyAction>& actions) {
        actions.clear();

        DummyAction action1;
        action1.action_ = -1.0;
        actions.push_back(action1);

        DummyAction action2;
        action1.action_ = 1.0;
        actions.push_back(action2);
      }
    }; //\ struct DummyAction
  }
}

#endif
