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
#include <util/types.hpp>

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
DEFINE_int32(max_iterations, 100,
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

// Flag for terminal state.
bool is_terminal = false;

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
  if (!is_terminal) {
    glutPostRedisplay();
    glutTimerFunc(FLAGS_refresh_rate, Timer, 0);
  }
}

// Reshape the window to maintain the correct aspect ratio.
void Reshape(GLsizei width, GLsizei height) {
  if (height == 0)
    height = 1;

  // Compute aspect ratio of the new window and for the grid.
  const GLfloat kWindowRatio =
    static_cast<GLfloat>(width) / static_cast<GLfloat>(height);
  const GLfloat kGridRatio =
    static_cast<GLfloat>(FLAGS_num_rows) / static_cast<GLfloat>(FLAGS_num_cols);

  // Set the viewport to cover the new window.
  glViewport(0, 0, width, height);

  // Set the clipping area.
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  // One of two possibilities:
  // (1) x_max = FLAGS_num_cols, y_max = height/width * FLAGS_num_cols
  //     is valid if y_max >= FLAGS_num_rows, otherwise
  // (2) y_max = FLAGS_num_rows, x_max = width/height * FLAGS_num_rows
  //     which is valid if x_max >= FLAGS_num_cols.

  // Try (1).
  const GLfloat y_max = static_cast<GLfloat>(FLAGS_num_cols) / kWindowRatio;
  if (y_max >= static_cast<GLfloat>(FLAGS_num_rows))
    gluOrtho2D(0.0, static_cast<GLfloat>(FLAGS_num_cols), 0.0, y_max);
  else {
    // Try (2).
    const GLfloat x_max = kWindowRatio * static_cast<GLfloat>(FLAGS_num_rows);
    CHECK_GE(x_max, static_cast<GLfloat>(FLAGS_num_cols));

    gluOrtho2D(0.0, x_max, 0.0, static_cast<GLfloat>(FLAGS_num_rows));
  }
}

// Take a single step.
void SingleIteration() {
  CHECK_NOTNULL(world);
  CHECK_NOTNULL(current_state);
  CHECK_NOTNULL(goal_state);
  CHECK_NOTNULL(solver);

  if (world->IsTerminal(*current_state))
    is_terminal = true;
  else {
    // Extract optimal policy.
    const DiscreteDeterministicPolicy<GridState, GridAction> policy =
      solver->Policy();

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
  const GLfloat kRadius = 0.5 - kEpsilon;

  current_state->Visualize(FLAGS_num_rows, kRadius, 0.0, 0.2, 0.8, 0.5);

  // Draw all previous states in a different color.
  for (const auto& state : history)
    state.Visualize(FLAGS_num_rows, kRadius, 0.0, 0.8, 0.2, 0.5);

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

  std::printf("Solver did not converge.\n");

  delete world;
  delete current_state;
  delete goal_state;
  return 1;
}
