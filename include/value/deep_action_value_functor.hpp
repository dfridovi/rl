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
// Defines the DeepActionValueFunctor class, which derives from the
// ContinuousActionValueFunctor base class. This class models the value function
// as a deep neural network using the Mininet DNN framework.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef RL_VALUE_LINEAR_ACTION_VALUE_FUNCTOR_H
#define RL_VALUE_LINEAR_ACTION_VALUE_FUNCTOR_H

#include <value/continuous_action_value_functor.hpp>
#include <util/types.hpp>

#include <mininet/net/network.h>
#include <mininet/layer/layer_params.h>
#include <miniet/loss/l2.h>

#include <random>
#include <vector>

using namespace mininet;

namespace rl {

  template<typename StateType, typename ActionType>
  class DeepActionValueFunctor :
    public ContinuousActionValueFunctor<StateType, ActionType> {
  public:
    // Constructor/destructor.
    ~DeepActionValueFunctor() {}
    explicit DeepActionValueFunctor(const std::vector<LayerParams>& layers,
                                    const LossFunctor::ConstPtr& loss,
                                    double momentum, double weight_decay);

    // Pure virtual method to output the value at a state/action pair.
    double operator()(const StateType& state, const ActionType& action) const;

    // Pure virtual method to do a gradient update to underlying weights.
    void Update(const StateType& state, const ActionType& action,
                double target, double step_size);

    // Choose an optimal action in the given state. Returns whether or not
    // optimization was successful.
    bool OptimalAction(const StateType& state, ActionType& action) const;

  private:
    // A deep network.
    Network net;

    // Training params.
    const double momentum_;
    const double weight_decay_;
  }; //\class DeepStateValueFunctor

// ----------------------------- IMPLEMENTATION ----------------------------- //

  template<typename StateType, typename ActionType>
  DeepActionValueFunctor<StateType, ActionType>::
  DeepActionValueFunctor(const std::vector<LayerParams>& layers,
                         const LossFunctor::ConstPtr& loss,
                         double momentum, double weight_decay)
    : net(layer_params, loss),
      momentum_(momentum),
      weight_decay_(weight_decay) {}

  // Pure virtual method to output the value at a state/action pair.
  template<typename StateType, typename ActionType>
  double DeepActionValueFunctor<StateType, ActionType>::
  operator()(const StateType& state, const ActionType& action) const {
    // Unpack state and action into a single feature vector.
    VectorXd features(StateType::FeatureDimension() +
                      ActionType::FeatureDimension());
    state.Features(features.head(StateType::FeatureDimension()));
    action.Features(features.tail(ActionType::FeatureDimension()));

    // Run through the net.
    VectorXd output(1);
    net(features, output);

    return output(0);
  }

  // Pure virtual method to do a gradient update to underlying weights.
  template<typename StateType, typename ActionType>
  void DeepActionValueFunctor<StateType, ActionType>::
  Update(const StateType& state, const ActionType& action,
         double target, double step_size) {
    // Convert state/action + target into input/output vectors.
    VectorXd input(StateType::FeatureDimension() +
                   ActionType::FeatureDimension());
    state.Features(input.head(StateType::FeatureDimension()));
    action.Features(input.tail(ActionType::FeatureDimension()));

    VectorXd output(1);
    output(0) = target;

    // Compute average layer inputs and deltas.
    std::vector<MatrixXd> derivatives;
    loss = network_.RunBatch(std::vector<VectorXd>({input}),
                             std::vector<VectorXd>({output}), derivatives);

    // Update weights.
    network_.UpdateWeights(derivatives, step_size, momentum_, weight_decay_);
  }

  // Choose an optimal action in the given state. Returns whether or not
  // optimization was successful.
  template<typename StateType, typename ActionType>
  bool DeepActionValueFunctor<StateType, ActionType>::
  OptimalAction(const StateType& state, ActionType& action) const {

  }
}  //\namespace rl

#endif
