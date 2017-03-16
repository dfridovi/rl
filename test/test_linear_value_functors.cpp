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
// Unit tests for linear value function approximators.
//
////////////////////////////////////////////////////////////////////////////////

#include <value/linear_state_value_functor.hpp>
#include <value/linear_action_value_functor.hpp>
#include <util/types.hpp>

#include "dummy_state_action.hpp"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <random>

using namespace rl;
using namespace rl::test;

// Test that a linear state value function converges to a true linear
// ground truth.
TEST(LinearStateValueFunctor, TestConvergence) {
  const double kEligibilityDecay = 0.0;
  const size_t kNumTrainingPoints = 10000;
  const size_t kNumTestingPoints = 100;
  const double kStepSize = 1e-2;
  const double kEpsilon = 1e-3;

  ContinousStateValue<DummyState>::Ptr value =
    LinearStateValue<DummyState>::Create(kEligibilityDecay);

  // Start a random number generator.
  std::random_device rd;
  std::default_random_engine rng(rd());
  std::uniform_real_distribution<double> unif(-1.0, 1.0);

  // Pick a random coefficient.
  const double state_coeff = unif(rng);

  // Iterate the specified number of times to train.
  for (size_t ii = 0; ii < kNumTrainingPoints; ii++) {
    DummyState random_state;

    value->Update(random_state, state_coeff * random_state.state_, kStepSize);
  }

  // Test the specified number of times.
  for (size_t ii = 0; ii < kNumTestingPoints; ii++) {
    DummyState random_state;

    EXPECT_NEAR(value(random_state),
                state_coeff * random_state.state_, kEpsilon);
  }
}

// Test that this action value functor's copy constructor is correct.
TEST(LinearStateValueFunctor, TestCopyConstructor) {
  const double kEligibilityDecay = 0.0;
  const size_t kNumTrainingPoints = 100;
  const size_t kNumChecks = 100;
  const double kStepSize = 1e-2;
  const double kEpsilon = 1e-3;

  ContinousStateValue<DummyState>::Ptr value =
    LinearStateValue<DummyState>::Create(kEligibilityDecay);

  // Create a const copy.
  const auto copy = value->Copy();

  // Try out a bunch of points on the original value functor.
  std::vector<DummyState> old_states;
  std::vector<double> old_values;

  for (size_t ii = 0; ii < kNumChecks; ii++) {
    DummyState random_state;

    old_states.push_back(random_state);
    old_values.push_back(value->Get(random_state));
  }

  // Start a random number generator.
  std::random_device rd;
  std::default_random_engine rng(rd());
  std::uniform_real_distribution<double> unif(-1.0, 1.0);

  // Pick a random coefficient.
  const double state_coeff = unif(rng);

  // Iterate the specified number of times to train.
  for (size_t ii = 0; ii < kNumTrainingPoints; ii++) {
    DummyState random_state;

    value->Update(random_state, state_coeff * random_state.state_, kStepSize);
  }

  // Test the specified number of times. For each stored state/action/value,
  // make sure that the copied value functor still matches the original before
  // it was trained.
  for (size_t ii = 0; ii < kNumChecks; ii++) {
    EXPECT_EQ(copy->Get(old_states[ii]), old_values[ii]);
  }

}

// Test that a linear action value function converges to a true linear
// ground truth.
TEST(LinearActionValueFunctor, TestConvergence) {
  const size_t kNumTrainingPoints = 10000;
  const size_t kNumTestingPoints = 100;
  const double kStepSize = 1e-2;
  const double kEpsilon = 1e-3;

  ContinousStateValue<DummyState>::Ptr value =
    LinearActionValue<DummyState>::Create();

  // Start a random number generator.
  std::random_device rd;
  std::default_random_engine rng(rd());
  std::uniform_real_distribution<double> unif(-1.0, 1.0);

  // Pick random coefficients.
  const double state_coeff = unif(rng);
  const double action_coeff = unif(rng);

  // Iterate the specified number of times to train.
  for (size_t ii = 0; ii < kNumTrainingPoints; ii++) {
    DummyState random_state;
    DummyAction random_action;

    const double result =
      state_coeff * random_state.state_ + action_coeff * random_action.action_;

    value->Update(std::vector<DummyState>({random_state}),
                  std::vector<DummyAction>({random_action}),
                  std::vector<double>({result}), kStepSize);
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

TEST(LinearActionValueFunctor, TestCopyConstructor) {
  const size_t kNumTrainingPoints = 100;
  const size_t kNumChecks = 100;
  const double kStepSize = 1e-2;
  const double kEpsilon = 1e-3;

  ContinousStateValue<DummyState>::Ptr value =
    LinearActionValue<DummyState>::Create();

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
  std::uniform_real_distribution<double> unif(-1.0, 1.0);

  // Pick random coefficients.
  const double state_coeff = unif(rng);
  const double action_coeff = unif(rng);

  // Iterate the specified number of times to train.
  for (size_t ii = 0; ii < kNumTrainingPoints; ii++) {
    DummyState random_state;
    DummyAction random_action;

    const double result =
      state_coeff * random_state.state_ + action_coeff * random_action.action_;

    value->Update(std::vector<DummyState>({random_state}),
                  std::vector<DummyAction>({random_action}),
                  std::vector<double>({result}), kStepSize);
  }

  // Test the specified number of times. For each stored state/action/value,
  // make sure that the copied value functor still matches the original before
  // it was trained.
  for (size_t ii = 0; ii < kNumChecks; ii++) {
    EXPECT_EQ(copy->Get(old_states[ii], old_actions[ii]), old_values[ii]);
  }
}
