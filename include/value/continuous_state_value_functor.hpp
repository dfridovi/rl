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
// Defines the ContinuousStateValueFunctor class, which derives from the
// StateValueFunctor base class.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef RL_VALUE_CONTINUOUS_STATE_VALUE_FUNCTOR_H
#define RL_VALUE_CONTINUOUS_STATE_VALUE_FUNCTOR_H

#include <value/state_value_functor.hpp>

#include <unordered_map>
#include <limits>
#include <vector>

namespace rl {

  template<typename StateType>
  struct ContinuousStateValue : public StateValue<StateType> {
    // Typedefs.
    typedef std::shared_ptr<ContinuousStateValue> Ptr;
    typedef std::shared_ptr<const ContinuousStateValue> ConstPtr;

    virtual ~ContinuousStateValue() {}

    // Must implement a deep copy.
    virtual Ptr Copy() const = 0;

    // Pure virtual method to output the value at a state.
    virtual double Get(const StateType& state) const = 0;

    // Pure virtual method to do a gradient update to underlying weights.
    virtual void Update(const StateType& state, double target,
                        double step_size) = 0;

  protected:
    explicit ContinuousStateValue()
      : StateValue<StateType>() {}
  }; //\struct ContinuousStateValue

}  //\namespace rl

#endif
