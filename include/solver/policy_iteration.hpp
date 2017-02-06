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
// Defines the ModifiedPolicyIteration class. Current implementation assumes
// discrete state and action spaces, and deterministic environments.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef RL_SOLVER_MODIFIED_POLICY_ITERATION_H
#define RL_SOLVER_MODIFIED_POLICY_ITERATION_H

namespace rl {

  template<typename StateType, typename ActionType>
  class ModifiedPolicyIteration {
  public:
    ~ModifiedPolicyIteration() {}

    // Initialize to a random policy. Pass in the number of value function
    // updates per iteration.
    explicit ModifiedPolicyIteration(size_t num_value_updates)
      : num_value_updates_(num_value_updates) {}

    // Solve the MDP defined by the given environment.
    void Solve(const DiscreteEnvironment<StateType, ActionType>& environment);

    // Update policy to be greedy with respect to the current state
    // value function V.
    void UpdatePolicy();

    // Single round of optimization of the state value function, i.e.
    // set V(s) <== V(environment(Pi(s)).
    void UpdateValueFunction();



  }; //\class ModifiedPolicyIteration

}  //\namespace rl

#endif
