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
// Defines the GaussianActionValueFunctor class, which derives from the
// ActionValueFunctor base class.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef RL_VALUE_GAUSSIAN_ACTION_VALUE_FUNCTOR_H
#define RL_VALUE_GAUSSIAN_ACTION_VALUE_FUNCTOR_H

#include <value/action_value_functor.hpp>
#include <util/types.hpp>

#include <Eigen/LDLT>
#include <limits>
#include <iostream>
#include <random>
#include <vector>
#include <math.h>

namespace rl {

  template<typename StateType, typename ActionType>
  class GaussianActionValueFunctor :
    public ActionValueFunctor<StateType, ActionType> {
    // Constructor/destructor.
    ~GaussianActionValueFunctor() {}
    explicit GaussianActionValueFunctor(size_t num_points, double regularizer,
                                        double noise_variance)
      : ContinuousActionValueFunctor<StateType, ActionType>() {}

    // Pure virtual method to output the value at a state/action pair.
    double operator()(const StateType& state, const ActionType& action) const;

    // Pure virtual method to do a gradient update to underlying weights.
    void Update(const StateType& state, const ActionType& action,
                double target, double step_size);

    // Choose an optimal action in the given state. Returns whether or not
    // optimization was successful.
    bool OptimalAction(const StateType& state, ActionType& action) const;

  private:
    // Training covariance matrix, with vectors of state/action features
    // training targets, and length scales.
    MatrixXd covariance_;
    std::vector<VectorXd> points_;
    VectorXd targets_;
    VectorXd squared_lengths_;

    // Stable Cholesky solver.
    Eigen::LDLT<MatrixXd> cholesky_;

    // Output of covariance.inv() * targets_. Stored for speed.
    VectorXd regressed_targets_;

    // Noise variance.
    double noise_variance_;

    // Regularization parameter to trade off mean with variance in the
    // choice of optimal action.
    double regularizer_;

    // Covariance kernel function.
    double Kernel(const VectorXd& x, const VectorXd& y) const;
  }; //\class GaussianActionValueFunctor

// ------------------------------ IMPLEMENTATION ---------------------------- //

  template<typename StateType, typename ActionType>
  GaussianActionValueFunctor<StateType, ActionType>::
  GaussianActionValueFunctor(size_t num_points, double regularizer,
                             double noise_variance)
    : regularizer_(regularizer),
      noise_variance_(noise_variance),
      covariance_(MatrixXd::Zero(num_points, num_points)),
      targets_(VectorXd::Zero(num_points)),
      regressed_targets_(VectorXd::Zero(num_points)),
      squared_lengths_(VectorXd::Zero(StateType::FeatureDimension() +
                                      ActionType::FeatureDimension())) {
    // TODO! Be sure to call cholesky_.compute()!
  }

  // Compute the expected value of the GP at this point.
  template<typename StateType, typename ActionType>
  double GaussianActionValueFunctor<StateType, ActionType>::
  operator()(const StateType& state, const ActionType& action) const {
    // Extract a compound feature vector.
    VectorXd features(squared_lengths_.size());
    state.Features(features.head(StateType::FeatureDimension()));
    action.Features(features.tail(ActionType::FeatureDimension()));

    // Accumulate the total inner product between the cross covariance
    // of this point and the training set.
    double output = 0.0;
    for (size_t ii = 0; ii < points_.size(); ii++) {
      const double cross_covariance = Kernel(points_[ii], features);
      output += cross_covariance * regressed_targets_(ii);
    }
  }

  // Update all parameters.
  template<typename StateType, typename ActionType>
  void GaussianActionValueFunctor<StateType, ActionType>::
  Update(const StateType& state, const ActionType& action,
         double target, double step_size) {
    // TODO!
  }

  // Compute the optimal action, where 'optimal' is the solution to the
  // following program:
  //                arg max mean(s, a) - reg * var(s, a)
  //                     a
  // Note that the sign of 'regularizer' will determine whether the control
  // is biased toward "safe" areas or optimistic exploration.
  template<typename StateType, typename ActionType>
  bool GaussianActionValueFunctor<StateType, ActionType>::
  OptimalAction(const StateType& state, ActionType& action) const {
    // TODO!
  }

  // Covariance kernel function.
  template<typename StateType, typename ActionType>
  double GaussianActionValueFunctor<StateType, ActionType>::
  Kernel(const VectorXd& x, const VectorXd& y) {
    CHECK_EQ(x.size(), squared_lengths_.size());
    CHECK_EQ(y.size(), squared_lengths_.size());

    return std::exp(-0.5 * x.dot(y.cwiseQuotient(squared_lengths_)));
  }
}  //\namespace rl

#endif
