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
// Unit tests for modified policy iteration.
//
////////////////////////////////////////////////////////////////////////////////

#include <solver/td_lambda.hpp>
#include <solver/td_lambda_params.hpp>
#include <environment/grid_world.hpp>
#include <environment/grid_state.hpp>
#include <environment/grid_action.hpp>

#include <glog/logging.h>
#include <gtest/gtest.h>

using namespace rl;

// Test that the iteration converges within a reasonable number of steps for
// a very small example grid world.
TEST(TdLambda, TestConvergence) {
  const double kDiscountFactor = 0.9;
  const double kLambda = 0.5;
  const double kAlpha = 0.5;
  const size_t kMaxIterations = 100;
  const size_t kNumRollouts = 1;
  const int kRolloutLength = -1;
  const double kInitialEpsilon = 0.5;

  // Create an environment.
  const size_t kNumRows = 5;
  const size_t kNumCols = 5;
  const GridState start(0, 0);
  const GridState goal(kNumRows - 1, kNumCols - 1);
  const GridWorld world(kNumRows, kNumCols, goal);

  // Create a solver.
  TdLambdaParams params;
  params.discount_factor_ = kDiscountFactor;
  params.lambda_ = kLambda;
  params.alpha_ = kAlpha;
  params.max_iterations_ = kMaxIterations;
  params.num_rollouts_ = kNumRollouts;
  params.rollout_length_ = kRolloutLength;
  params.initial_epsilon_ = kInitialEpsilon;

  TdLambda<GridState, GridAction> solver(start, params);

  // Solve.
  EXPECT_TRUE(solver.Solve(world));
}

// Test that the iteration converges to the right answer within a
// reasonable number of steps for a very small example grid world.
TEST(TdLambda, TestConvergenceToOptimum) {
  const double kDiscountFactor = 0.9;
  const double kLambda = 0.5;
  const double kAlpha = 0.5;
  const size_t kMaxIterations = 100;
  const size_t kNumRollouts = 1;
  const int kRolloutLength = -1;
  const double kInitialEpsilon = 0.5;

  // Create an environment.
  const size_t kNumRows = 1;
  const size_t kNumCols = 5;
  const GridState start(0, 0);
  const GridState goal(kNumRows - 1, kNumCols - 1);
  const GridWorld world(kNumRows, kNumCols, goal);

  // Create a solver.
  TdLambdaParams params;
  params.discount_factor_ = kDiscountFactor;
  params.lambda_ = kLambda;
  params.alpha_ = kAlpha;
  params.max_iterations_ = kMaxIterations;
  params.num_rollouts_ = kNumRollouts;
  params.rollout_length_ = kRolloutLength;
  params.initial_epsilon_ = kInitialEpsilon;

  TdLambda<GridState, GridAction> solver(start, params);

  // Solve.
  EXPECT_TRUE(solver.Solve(world));

  // Check that we converged to the right policy.
  const DiscreteEpsilonGreedyPolicy<GridState, GridAction> policy =
    solver.Policy();

  for (size_t jj = 0; jj < kNumCols - 1; jj++) {
    GridAction action;
    EXPECT_TRUE(policy.Act(world, GridState(0, jj), action));
    EXPECT_EQ(action, GridAction(GridAction::Direction::RIGHT));
  }
}
