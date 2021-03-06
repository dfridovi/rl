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

#include "../value/continuous_action_value_functor.hpp"
#include "../value/gaussian_params.hpp"
#include "util/types.hpp"

#include <Eigen/Cholesky>
#include <glog/logging.h>
#include <limits>
#include <iostream>
#include <random>
#include <vector>
#include <math.h>

namespace rl {

  template<typename StateType, typename ActionType>
  class GaussianActionValue :
    public ContinuousActionValue<StateType, ActionType> {
  public:
    ~GaussianActionValue() {}

    // Factory method.
    static typename ContinuousActionValue<StateType, ActionType>::Ptr
    Create(const GaussianParams& params);

    // Must implement a deep copy.
    typename ContinuousActionValue<StateType, ActionType>::Ptr Copy() const;

    // Pure virtual method to output the value at a state/action pair.
    double Get(const StateType& state, const ActionType& action) const;

    // Pure virtual method to do a gradient update to underlying weights.
    // Returns average loss.
    double Update(const std::vector<StateType>& states,
                  const std::vector<ActionType>& actions,
                  const std::vector<double>& targets,
                  double step_size);

    // Choose an optimal action in the given state. Returns whether or not
    // optimization was successful.
    bool OptimalAction(const StateType& state, ActionType& action) const;

  private:
    explicit GaussianActionValue(const GaussianParams& params);

    // Covariance kernel function.
    double Kernel(const VectorXd& x, const VectorXd& y) const;

    // Compute the cross covariance vector of a feature vector with
    // the training data.
    void CrossCovariance(const VectorXd& features, VectorXd& cross) const;

    // Helper function to evaluate the GP.
    void Evaluate(const StateType& state, const ActionType& action,
                  double& mean, double& variance) const;

    // Training covariance matrix, with vectors of state/action features
    // training means, and length scales.
    MatrixXd covariance_;
    std::vector<VectorXd> points_;
    VectorXd means_;
    VectorXd lengths_;
    VectorXd squared_lengths_;

    // Fast Cholesky solver.
    Eigen::LLT<MatrixXd> cholesky_;

    // Output of covariance.inv() * means_. Stored for speed.
    VectorXd regressed_means_;

    // Noise variance.
    double noise_variance_;

    // Regularization parameter to trade off mean with variance in the
    // choice of optimal action.
    double regularizer_;

    // Max number of gradient steps to take for choosing the optimal action,
    // with given step size starting from one of 'num_inits_' random initial
    // points. 'epsilon_' is convergence criterion for gradient size.
    const size_t num_inits_;
    const size_t max_steps_;
    const double step_size_;
    const double epsilon_;
  }; //\class GaussianActionValue

// ------------------------------ IMPLEMENTATION ---------------------------- //

  // Factory method.
  template<typename StateType, typename ActionType>
  typename ContinuousActionValue<StateType, ActionType>::Ptr
  GaussianActionValue<StateType, ActionType>::
  Create(const GaussianParams& params) {
    typename ContinuousActionValue<StateType, ActionType>::Ptr
      ptr(new GaussianActionValue<StateType, ActionType>(params));
    return ptr;
  }

  // Must implement a deep copy.
  template<typename StateType, typename ActionType>
  typename ContinuousActionValue<StateType, ActionType>::Ptr
  GaussianActionValue<StateType, ActionType>::Copy() const {
    typename ContinuousActionValue<StateType, ActionType>::Ptr
      ptr(new GaussianActionValue<StateType, ActionType>(*this));
    return ptr;
  }

  // Constructor.
  template<typename StateType, typename ActionType>
  GaussianActionValue<StateType, ActionType>::
  GaussianActionValue(const GaussianParams& params)
    : regularizer_(params.regularizer_),
      noise_variance_(params.noise_variance_),
      max_steps_(params.max_steps_),
      num_inits_(params.num_inits_),
      step_size_(params.step_size_),
      epsilon_(params.epsilon_),
      covariance_(MatrixXd::Zero(params.num_points_, params.num_points_)),
      means_(VectorXd::Zero(params.num_points_)),
      regressed_means_(VectorXd::Zero(params.num_points_)),
      lengths_(params.lengths_),
      squared_lengths_(params.lengths_.cwiseProduct(params.lengths_)),
      ContinuousActionValue<StateType, ActionType>() {
    // Pick random points in the space for training.
    for (size_t ii = 0; ii < params.num_points_; ii++) {
      const StateType state;
      const ActionType action;

      // Unpack into a feature vector.
      VectorXd state_features(StateType::FeatureDimension());
      state.Features(state_features);

      VectorXd action_features(ActionType::FeatureDimension());
      action.Features(action_features);

      VectorXd features(state_features.size() + action_features.size());
      features.head(state_features.size()) = state_features;
      features.tail(action_features.size()) = action_features;

      // Add to list of training points.
      points_.push_back(features);
    }

    // Compute training covariance.
    for (size_t ii = 0; ii < points_.size(); ii++) {
      for (size_t jj = 0; jj < ii; jj++) {
        covariance_(ii, jj) = Kernel(points_[ii], points_[jj]);
        covariance_(jj, ii) = covariance_(ii, jj);
      }

      covariance_(ii, ii) = 1.0 + noise_variance_;
    }

    // Randomize training targets.
    std::random_device rd;
    std::default_random_engine rng(rd());
    std::normal_distribution<double> gaussian(0.0, 0.1);

    for (size_t ii = 0; ii < means_.size(); ii++)
      means_(ii) = gaussian(rng);

    // Compute Cholesky decomposition of 'covariance_' for quick solving.
    cholesky_ = covariance_.llt();

    // Set 'regressed_means_' for speed.
    regressed_means_ = cholesky_.solve(means_);
  }

  // Helper function to evaluate the GP.
  template<typename StateType, typename ActionType>
  void GaussianActionValue<StateType, ActionType>::
  Evaluate(const StateType& state, const ActionType& action,
           double& mean, double& variance) const {
    // Compute cross covariance vector.
    VectorXd features(StateType::FeatureDimension() +
                      ActionType::FeatureDimension());
    this->Unpack(state, action, features);

    VectorXd cross(points_.size());
    CrossCovariance(features, cross);

    // Regress the cross covariance on the training covariance.
    const VectorXd regressed_cross = cholesky_.solve(cross);

    // Compute the total inner product between the cross covariance
    // of this point and the training set.
    mean = cross.dot(regressed_means_);
    variance = 1.0 - cross.dot(regressed_cross);
  }

  // Compute the expected value of the GP at this point.
  template<typename StateType, typename ActionType>
  double GaussianActionValue<StateType, ActionType>::
  Get(const StateType& state, const ActionType& action) const {
    // Compute cross covariance vector.
    VectorXd features(StateType::FeatureDimension() +
                      ActionType::FeatureDimension());
    this->Unpack(state, action, features);

    VectorXd cross(points_.size());
    CrossCovariance(features, cross);

    // Compute the total inner product between the cross covariance
    // of this point and the training set.
    return cross.dot(regressed_means_);
  }

  // Update all parameters. Return average loss.
  template<typename StateType, typename ActionType>
  double GaussianActionValue<StateType, ActionType>::
  Update(const std::vector<StateType>& states,
         const std::vector<ActionType>& actions,
         const std::vector<double>& targets, double step_size) {
    CHECK_EQ(states.size(), actions.size());
    CHECK_EQ(states.size(), targets.size());

    // Iterate over each state/action pair and average the gradients.
    double loss = 0.0;
    VectorXd gradient = VectorXd::Zero(means_.size());
    VectorXd features(StateType::FeatureDimension() +
                      ActionType::FeatureDimension());
    VectorXd cross(points_.size());
    for (size_t ii = 0; ii < states.size(); ii++) {
      // Compute cross covariance.
      this->Unpack(states[ii], actions[ii], features);
      CrossCovariance(features, cross);

      // Compute the gradient.
      const VectorXd regressed = cholesky_.solve(cross);
      const double error = regressed.dot(means_) - targets[ii];

      // Catch nan. Set to zero.
      if (isnan(error)) {
        LOG(WARNING) << "GaussianActionValue: Error was nan. Skipping.";
        continue;
      }

      //      std::printf("Error was: %f = %f - %f\n",
      //                  error, error + targets[ii], targets[ii]);

      loss += error * error;
      gradient += error * regressed;
    }

    // Do a gradient update.
    means_ -= step_size * gradient / static_cast<double>(states.size());

    // Update 'regressed_means_' for speed later.
    regressed_means_ = cholesky_.solve(means_);

    // Return loss. Divide by two for consistency.
    return 0.5 * loss / static_cast<double>(states.size());
  }

  // Compute the optimal action, where 'optimal' is the solution to the
  // following program:
  //                arg max mean(s, a) + reg * var(s, a)
  //                     a
  // Note that the sign of 'regularizer' will determine whether the control
  // is biased toward "safe" areas or optimistic exploration. In practice, this
  // problem is usually non-convex, and we solve it by taking several gradient
  // steps from a few random initializations and returning the best final value.
  template<typename StateType, typename ActionType>
  bool GaussianActionValue<StateType, ActionType>::
  OptimalAction(const StateType& state, ActionType& action) const {
    // Get a list of possible actions.
    std::vector<ActionType> candidates;
    ActionType::DiscreteValues(candidates);

    // Find the best option in this list.
    double max_value = kInvalidValue;
    for (const auto& candidate : candidates) {
      double mean, variance;
      Evaluate(state, candidate, mean, variance);

      const double value = mean + regularizer_ * variance;

      if (value > max_value) {
        max_value = value;
        action = candidate;
      }
    }

    return (max_value != kInvalidValue);

#if 0
    VectorXd features(StateType::FeatureDimension() +
                      ActionType::FeatureDimension());

    // Try a few random initial actions and keep the best one
    // after running a few iterations of gradient ascent.
    double best_value = kInvalidReward;
    bool has_converged = false;

    VectorXd cross(points_.size());
    VectorXd regressed_cross(points_.size());
    MatrixXd Jt(ActionType::FeatureDimension(), points_.size());

    for (size_t ii = 0; ii < num_inits_; ii++) {
      // Start from a random initial action.
      ActionType current_action;
      this->Unpack(state, current_action, features);

      // Run a few steps of gradient ascent to adjust this action.
      has_converged = false;
      for (size_t jj = 0; jj < max_steps_; jj++) {
        // Compute the cross covariance of this state-action pair with the data.
        CrossCovariance(features, cross);

        // Compute the Jacobian transpose of cross covariance with respect
        // to feature vector.
        for (size_t jj = 0; jj < points_.size(); jj++) {
          const VectorXd action_diff =
            points_[jj].tail(ActionType::FeatureDimension()) -
            features.tail(ActionType::FeatureDimension());

          Jt.col(jj) = cross(jj) * action_diff.cwiseQuotient(
            squared_lengths_.tail(ActionType::FeatureDimension()));
        }

        // Compute the intermediate derivative with respect to cross covariance.
        regressed_cross = cholesky_.solve(cross);
        const VectorXd cross_gradient =
          regressed_means_ + regularizer_ * regressed_cross;

        // Compute the gradient with respect to the action feature vector.
        const VectorXd gradient = Jt * cross_gradient;

        // Gradient update.
        features.tail(ActionType::FeatureDimension()) += step_size_ * gradient;

        // Check convergence.
        if (gradient.norm() < epsilon_) {
          has_converged = true;
          break;
        }
      }

      // Check if this value is best.
      const double value =
        (regressed_means_ + regularizer_ * regressed_cross).dot(cross);

      if (value > best_value) {
        best_value = value;
        action.FromFeatures(features.tail(ActionType::FeatureDimension()));
      }
    }

    return has_converged;
#endif
  }

  // Covariance kernel function.
  template<typename StateType, typename ActionType>
  double GaussianActionValue<StateType, ActionType>::
  Kernel(const VectorXd& x, const VectorXd& y) const {
    CHECK_EQ(x.size(), lengths_.size());
    CHECK_EQ(y.size(), lengths_.size());

    const VectorXd normalized_delta = (x - y).cwiseQuotient(lengths_);
    return std::exp(-0.5 * normalized_delta.squaredNorm());
  }

  // Compute the cross covariance vector of a feature vector with
  // the training data.
  template<typename StateType, typename ActionType>
  void GaussianActionValue<StateType, ActionType>::
  CrossCovariance(const VectorXd& features, VectorXd& cross) const {
    CHECK_EQ(cross.size(), points_.size());
    CHECK_EQ(features.size(),
             StateType::FeatureDimension() + ActionType::FeatureDimension());

    // Compute cross covariance of features with training points.
    for (size_t ii = 0; ii < points_.size(); ii++)
      cross(ii) = Kernel(features, points_[ii]);
  }

}  //\namespace rl

#endif
