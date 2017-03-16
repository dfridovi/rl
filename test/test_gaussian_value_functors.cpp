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
// Unit tests for gaussian value function approximators.
//
////////////////////////////////////////////////////////////////////////////////

#include <value/gaussian_action_value_functor.hpp>
#include <value/experience_replay.hpp>
#include <util/types.hpp>

#include "dummy_state_action.hpp"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <random>

using namespace rl;
using namespace rl::test;

// Test that a gaussian action value function converges to a true linear
// ground truth.
TEST(GaussianActionValueFunctor, TestConvergence) {
  const size_t kNumTrainingPoints = 1000;
  const size_t kNumTestingPoints = 10;
  const size_t kNumGpPoints = 20;
  const double kStepSize = 0.01;
  double kLearningRate = 0.5;
  const double kLearningRateDecay = 0.9;
  const double kEpsilon = 0.1;
  const double kRegularizer = 0.0;
  const double kNoiseVariance = 0.01;
  const size_t kNumInits = 1;
  const size_t kMaxSteps = 10;
  const double kLengthScale = 1.0;
  const size_t kBatchSize = 5;
  const size_t kNumUpdates = 1000;

  const VectorXd kLengths =
    VectorXd::Constant(DummyState::FeatureDimension() +
                       DummyAction::FeatureDimension(), kLengthScale);

  // Create action value functor.
  GaussianParams params;
  params.num_points_ = kNumGpPoints;
  params.noise_variance_ = kNoiseVariance;
  params.regularizer_ = kRegularizer;
  params.step_size_ = kStepSize;
  params.max_steps_ = kMaxSteps;
  params.num_inits_ = kNumInits;
  params.epsilon_ = kEpsilon;
  params.lengths_ = kLengths;

  ContinuousActionValue<DummyState, DummyAction>::Ptr value =
    GaussianActionValue<DummyState, DummyAction>::Create(params);

  // Start a random number generator.
  std::random_device rd;
  std::default_random_engine rng(rd());
  std::uniform_real_distribution<double> unif(-0.5, 0.5);

  // Pick random coefficients.
  const double state_coeff = unif(rng);
  const double action_coeff = unif(rng);

  // Generate training data.
  ExperienceReplay<DummyState, DummyAction> replay;
  for (size_t ii = 0; ii < kNumTrainingPoints; ii++) {
    DummyState random_state;
    DummyAction random_action;

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
    const double loss = value->Update(states, actions, targets, kLearningRate);

    if (ii % 100 == 0) {
      //      std::printf("Loss at interation %zu = %f.\n", ii, loss);
      kLearningRate *= kLearningRateDecay;
    }
  }

  // Test the specified number of times.
  for (size_t ii = 0; ii < kNumTestingPoints; ii++) {
    DummyState random_state;
    DummyAction random_action;

    const double result =
      state_coeff * random_state.state_ + action_coeff * random_action.action_;

    EXPECT_NEAR(value->Get(random_state, random_action), result, kEpsilon);
  }
}

// Test that this action value functor's copy constructor is correct.
TEST(GaussianActionValueFunctor, TestCopyConstructor) {
  const size_t kNumChecks = 100;
  const size_t kNumTrainingPoints = 100;
  const size_t kNumGpPoints = 20;
  const double kStepSize = 0.01;
  const double kLearningRate = 0.5;
  const double kEpsilon = 0.1;
  const double kRegularizer = 0.0;
  const double kNoiseVariance = 0.01;
  const size_t kNumInits = 1;
  const size_t kMaxSteps = 10;
  const double kLengthScale = 1.0;
  const size_t kBatchSize = 5;
  const size_t kNumUpdates = 100;

  const VectorXd kLengths =
    VectorXd::Constant(DummyState::FeatureDimension() +
                       DummyAction::FeatureDimension(), kLengthScale);

  // Create action value functor.
  GaussianParams params;
  params.num_points_ = kNumGpPoints;
  params.noise_variance_ = kNoiseVariance;
  params.regularizer_ = kRegularizer;
  params.step_size_ = kStepSize;
  params.max_steps_ = kMaxSteps;
  params.num_inits_ = kNumInits;
  params.epsilon_ = kEpsilon;
  params.lengths_ = kLengths;

  ContinuousActionValue<DummyState, DummyAction>::Ptr value =
    GaussianActionValue<DummyState, DummyAction>::Create(params);

  // Create a const copy.
  const auto copy = value->Copy();

  // Try out a bunch of points on the original value functor.
  std::vector<DummyState> old_states;
  std::vector<DummyAction> old_actions;
  std::vector<double> old_values;

  for (size_t ii = 0; ii < kNumChecks; ii++) {
    DummyState random_state;
    DummyAction random_action;

    old_states.push_back(random_state);
    old_actions.push_back(random_action);
    old_values.push_back(value->Get(random_state, random_action));
  }

  // Start a random number generator.
  std::random_device rd;
  std::default_random_engine rng(rd());
  std::uniform_real_distribution<double> unif(-0.5, 0.5);

  // Pick random coefficients.
  const double state_coeff = unif(rng);
  const double action_coeff = unif(rng);

  // Generate training data.
  ExperienceReplay<DummyState, DummyAction> replay;
  for (size_t ii = 0; ii < kNumTrainingPoints; ii++) {
    DummyState random_state;
    DummyAction random_action;

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
    const double loss = value->Update(states, actions, targets, kLearningRate);
  }

  // Test the specified number of times. For each stored state/action/value,
  // make sure that the copied value functor still matches the original before
  // it was trained.
  for (size_t ii = 0; ii < kNumChecks; ii++) {
    EXPECT_EQ(copy->Get(old_states[ii], old_actions[ii]), old_values[ii]);
  }
}
