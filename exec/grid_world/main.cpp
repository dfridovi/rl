/*
 * Copyright (c) 2015, The Regents of the University of California (Regents).
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

#include <environment/grid_world.hpp>
#include <environment/grid_action.hpp>
#include <environment/grid_state.hpp>
#include <solver/modified_policy_iteration.hpp>
#include <util/types.h>

#include <glog/logging.h>
#include <gflags/gflags.h>
#include <iostream>
#include <random>
#include <math.h>

#ifdef SYSTEM_OSX
#include <GLUT/glut.h>
#endif

#ifdef SYSTEM_LINUX
#include <GL/glut.h>
#endif

using namespace rl;

DEFINE_int32(refresh_rate, 1000, "Refresh rate in milliseconds.");
DEFINE_int32(num_value_updates, 1,
             "Number of value updates per policy iteration.");
DEFINE_int32(max_iterations, 10,
             "Maximum umber of iterations to run modified policy iteration.");
DEFINE_double(discount_factor, 0.9, "Discount factor.");
DEFINE_int32(num_rows, 5, "Number of rows in the grid.");
DEFINE_int32(num_cols, 5, "Number of columns in the grid.");

// Create a globally-defined GridWorld, as well as current and goal states.
GridWorld* world = NULL;
GridState* current_state = NULL;
GridState* goal_state = NULL;

// History of past states.
std::vector<GridState> history;

// Create a solver.
ModifiedPolicyIteration<GridState, GridAction>* solver = NULL;

// Step counter.
size_t step_counter = 0;

// Initialize OpenGL.
void InitGL() {
  // Set the "clearing" or background color as black/opaque.
  glClearColor(0.0, 0.0, 0.0, 1.0);

  // Set up alpha blending.
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_BLEND);
}

// Timer callback. Re-render at the specified rate.
void Timer(int value) {
  glutPostRedisplay();
  glutTimerFunc(FLAGS_refresh_rate, Timer, 0);
}

// Reshape the window to maintain the correct aspect ratio.
void Reshape(GLsizei width, GLsizei height) {
  if (height == 0)
    height = 1;

  // Compute aspect ratio fo the new window and for the grid.
  const GLfloat kWindowRatio =
    static_cast<GLfloat>(width) / static_cast<GLfloat>(height);
  const GLfloat kGridRatio =
    static_cast<GLfloat>(FLAGS_num_rows) / static_cast<GLfloat>(FLAGS_num_cols);

  // Set the viewport to cover the new window.
  glViewport(0, 0, width, height);

  // Set the clipping area to be a square in the positive quadrant.
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  if (width >= height) {
    // Larger window width than height.
    if (kGridRatio >= kWindowRatio) {
      // Grid is wider than window.
      gluOrtho2D(0.0, static_cast<GLfloat>(FLAGS_num_rows),
                 0.0, (static_cast<GLfloat>(FLAGS_num_cols) *
                       kGridRatio / kWindowRatio));
    } else {
      // Window is wider than grid.
      gluOrtho2D(0.0, (static_cast<GLfloat>(FLAGS_num_rows) *
                       kWindowRatio / kGridRatio),
                 0.0, static_cast<GLfloat>(FLAGS_num_cols));
    }
  } else {
    // Larger height than width.
    if (kGridRatio <= kWindowRatio) {
      // Grid is narrower than window.
      gluOrtho2D(0.0, (static_cast<GLfloat>(FLAGS_num_rows) *
                       kWindowRatio / kGridRatio),
                 0.0, static_cast<GLfloat>(FLAGS_num_cols));
    } else {
      // Window is narrower than grid.
      gluOrtho2D(0.0, static_cast<GLfloat>(FLAGS_num_rows),
                 0.0, (static_cast<GLfloat>(FLAGS_num_cols) *
                       kGridRatio / kWindowRatio));
    }
  }
}

// Take a single step.
void SingleIteration() {
  CHECK_NOTNULL(world);
  CHECK_NOTNULL(current_state);
  CHECK_NOTNULL(goal_state);
  CHECK_NOTNULL(solver);

  // Extract optimal policy and value function.
  const DiscreteDeterministicPolicy<GridState, GridAction> policy =
    solver->Policy();
  const DiscreteStateValueFunctor<GridState> value = solver->Value();

  // Only move if not in the goal state.
  if (*current_state != *goal_state) {
    const GridState copy_state(current_state->ii_, current_state->jj_);
    history.push_back(copy_state);

    // Get optimal action.
    GridAction action;
    policy.Act(*current_state, action);

    // Simulate this action.
    const double reward = world->Simulate(*current_state, action);

    // Print out step counter and reward.
    std::printf("Received reward of %f on step %zu.\n", reward, ++step_counter);
  }

  // Visualize no matter what.
  world->Visualize();

  // Draw the current state as a circle (polygon with a ton of sides).
  const GLfloat kEpsilon = 0.02;
  const GLfloat current_x = static_cast<GLfloat>(current_state->jj_) + 0.5;
  const GLfloat current_y =
    static_cast<GLfloat>(FLAGS_num_rows - current_state->ii_) - 0.5;

  const size_t kNumVertices = 100;
  const GLfloat kRadius = 0.5 - kEpsilon;

  glBegin(GL_POLYGON);
  glColor4f(0.0, 0.2, 0.8, 0.5);
  for (size_t ii = 0; ii < kNumVertices; ii++) {
    const GLfloat angle = 2.0 * M_PI *
      static_cast<GLfloat>(ii) / static_cast<GLfloat>(kNumVertices);
    glVertex2f(current_x + kRadius * cos(angle),
               current_y + kRadius * sin(angle));
  }
  glEnd();

  // Draw all previous states in a different color.
  for (const auto& state : history) {
    const GLfloat past_x = static_cast<GLfloat>(state.jj_) + 0.5;
    const GLfloat past_y =
      static_cast<GLfloat>(FLAGS_num_rows - state.ii_) - 0.5;

    glBegin(GL_POLYGON);
    glColor4f(0.0, 0.8, 0.2, 0.5);
    for (size_t ii = 0; ii < kNumVertices; ii++) {
      const GLfloat angle = 2.0 * M_PI *
        static_cast<GLfloat>(ii) / static_cast<GLfloat>(kNumVertices);
      glVertex2f(past_x + kRadius * cos(angle),
                 past_y + kRadius * sin(angle));
    }
    glEnd();
  }

  // Swap buffers.
  glutSwapBuffers();
}

// Set everything up and go!
int main(int argc, char** argv) {
  // Set up logging.
  google::InitGoogleLogging(argv[0]);

  // Parse flags.
  google::ParseCommandLineFlags(&argc, &argv, true);
  CHECK_GE(FLAGS_num_rows, 1);
  CHECK_GE(FLAGS_num_cols, 1);

  // Create initial and goal states at opposite corners.
  goal_state = new GridState(FLAGS_num_rows - 1, FLAGS_num_cols - 1);
  current_state = new GridState(0, 0);

  // Set up grid world.
  world = new GridWorld(FLAGS_num_rows, FLAGS_num_cols, *goal_state);

  // Set up the solver.
  solver = new ModifiedPolicyIteration<GridState, GridAction>(
     FLAGS_num_value_updates, FLAGS_max_iterations, FLAGS_discount_factor);

  if (solver->Solve(*world)) {
    // Set up OpenGL window.
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE);
    glutInitWindowSize(320, 320);
    glutInitWindowPosition(50, 50);
    glutCreateWindow("Grid World");
    glutDisplayFunc(SingleIteration);
    glutReshapeFunc(Reshape);
    glutTimerFunc(0, Timer, 0);
    InitGL();
    glutMainLoop();

    delete world;
    delete current_state;
    delete goal_state;
    return 0;
  }

  delete world;
  delete current_state;
  delete goal_state;
  return 1;
}
