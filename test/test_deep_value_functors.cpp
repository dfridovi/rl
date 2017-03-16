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
// Unit tests for deep value function approximators.
//
////////////////////////////////////////////////////////////////////////////////

#include <value/deep_action_value_functor.hpp>
#include <value/experience_replay.hpp>
#include <util/types.hpp>

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <random>

using namespace rl;
using namespace mininet;

// Dummy scalar state and action structs.
struct DummyState {
  double state_;

  static constexpr size_t FeatureDimension() { return 1; }
  void Features(VectorXd& features) const {
    CHECK_EQ(features.size(), FeatureDimension());
    features(0) = state_;
  }
}; //\ struct DummyState

struct DummyAction {
  double action_;

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

// Test that a deep action value function converges to a true linear
// ground truth.
TEST(DeepActionValueFunctor, TestConvergence) {
  const size_t kNumTrainingPoints = 100;
  const size_t kNumTestingPoints = 10;
  const double kStepSize = 1e-2;
  const double kEpsilon = 1e-2;
  const double kMomentum = 0.0;
  const double kWeightDecay = 0.0;
  const size_t kBatchSize = 5;
  const size_t kNumUpdates = 10000;

  // Construct layers.
  std::vector<LayerParams> layers;
  layers.push_back(LayerParams(LINEAR,
                               DummyState::FeatureDimension() +
                               DummyAction::FeatureDimension(),
                               1));

  // Construct loss.
  LossFunctor::ConstPtr loss = L2::Create();

  // Create action value functor.
  DeepActionValueFunctor<DummyState, DummyAction> value(layers, loss, kMomentum,
                                                        kWeightDecay);

  // Start a random number generator.
  std::random_device rd;
  std::default_random_engine rng(rd());
  std::uniform_real_distribution<double> unif(-1.0, 1.0);

  // Pick random coefficients.
  const double state_coeff = unif(rng);
  const double action_coeff = unif(rng);

  // Generate training data.
  ExperienceReplay<DummyState, DummyAction> replay;
  for (size_t ii = 0; ii < kNumTrainingPoints; ii++) {
    DummyState random_state;
    random_state.state_ = unif(rng);

    DummyAction random_action;
    random_action.action_ = unif(rng);

    const double result =
      state_coeff * random_state.state_ + action_coeff * random_action.action_;

    replay.Add(random_state, random_action, result, random_state);
  }

  // Iterate the specified number of times to train.
  std::vector<DummyState> states, next_states;
  std::vector<DummyAction> actions;
  std::vector<double> targets;
  for (size_t ii = 0; ii < kNumUpdates; ii++) {
    // Get a batch.
    ASSERT_TRUE(replay.Sample(kBatchSize, states, actions,
                              targets, next_states));

    // Update.
    value.Update(states, actions, targets, kStepSize);
  }

  // Test the specified number of times.
  for (size_t ii = 0; ii < kNumTestingPoints; ii++) {
    DummyState random_state;
    random_state.state_ = unif(rng);

    DummyAction random_action;
    random_action.action_ = unif(rng);

    const double result =
      state_coeff * random_state.state_ + action_coeff * random_action.action_;

    EXPECT_NEAR(value(random_state, random_action), result, kEpsilon);
  }
}
