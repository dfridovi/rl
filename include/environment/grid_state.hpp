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
// Defines the GridState struct, which is basically just a (ii, jj) pair.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef RL_ENVIRONMENT_GRID_STATE_H
#define RL_ENVIRONMENT_GRID_STATE_H

#include <boost/functional/hash.hpp>
#include <stddef.h>
#include <math.h>

#ifdef SYSTEM_OSX
#include <GLUT/glut.h>
#endif

#ifdef SYSTEM_LINUX
#include <GL/glut.h>
#endif

namespace rl {

  struct GridState {
    // Position.
    size_t ii_;
    size_t jj_;

    // Constructor/destructor.
    ~GridState() {}
    GridState(size_t ii, size_t jj)
      : ii_(ii), jj_(jj) {}

    // (In)equality operators.
    bool operator==(const GridState& rhs) const {
      return ii_ == rhs.ii_ && jj_ == rhs.jj_;
    }

    bool operator!=(const GridState& rhs) const {
      return ii_ != rhs.ii_ || jj_ != rhs.jj_;
    }

    // OpenGL visualization.
    void Visualize(size_t num_rows, double radius,
                   double r, double g, double b, double a) const {
      const size_t kNumVertices = 100;

      // Extract current position on xy plane.
      const GLfloat current_x = static_cast<GLfloat>(jj_) + 0.5;
      const GLfloat current_y =
        static_cast<GLfloat>(num_rows - ii_) - 0.5;

      glBegin(GL_POLYGON);
      glColor4f(r, g, b, a);
      for (size_t ii = 0; ii < kNumVertices; ii++) {
        const GLfloat angle = 2.0 * M_PI *
          static_cast<GLfloat>(ii) / static_cast<GLfloat>(kNumVertices);
        glVertex2f(current_x + radius * cos(angle),
                   current_y + radius * sin(angle));
      }
      glEnd();
    }

    // Hash functor.
    struct Hash {
      size_t operator()(const GridState& state) const {
        size_t seed = 0;
        boost::hash_combine(seed, boost::hash_value(state.ii_));
        boost::hash_combine(seed, boost::hash_value(state.jj_));

        return seed;
      }
    }; //\struct Hash
  }; //\struct GridState

}  //\namespace rl

#endif
