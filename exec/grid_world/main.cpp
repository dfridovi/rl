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

DEFINE_int32(refresh_rate, 1000, "Refresh rate in milliseconds.");
DEFINE_bool(iterate_forever, false, "Iterate ad inifinitum?");
DEFINE_int32(num_iterations, 10, "Number of iterations to run exploration.");
DEFINE_int32(num_rows, 5, "Number of rows in the grid.");
DEFINE_int32(num_cols, 5, "Number of columns in the grid.");

// Create a globally-defined GridWorld, as well as current and goal states.
rl::GridWorld* world = NULL;
rl::GridState* current_state = NULL;
rl::GridState* goal_state = NULL;

// Create a globally-defined step counter.
unsigned int step_count = 0;

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

  // Take an action if not in terminal state or have steps left.
  if (*current_state != *goal_state &&
      (FLAGS_iterate_forever || step_count < FLAGS_num_iterations)) {
    // Pick a random action.
    while (!world->Simulate(*current_state, rl::GridAction()));

    // Increment step counter.
    step_count++;
  }

  // Visualize no matter what.
  glClear(GL_COLOR_BUFFER_BIT);
  const GLfloat kEpsilon = 0.02;

  // Display each grid cell as a GL_QUAD, centered at the appropriate
  // location, with a small 'epsilon' fudge factor between cells. Color
  // the goal state red.
  glBegin(GL_QUADS);
  for (size_t ii = 0; ii < FLAGS_num_rows; ii++) {
    for (size_t jj = 0; jj < FLAGS_num_cols; jj++) {
      // Check for goal state.
      if (ii == goal_state->ii_ && jj == goal_state->jj_)
        glColor4f(0.8, 0.0, 0.2, 0.9);
      else
        glColor4f(0.9, 0.9, 0.9, 0.9);

      // Convert to top left x, y coords.
      const GLfloat x = static_cast<GLfloat>(jj);
      const GLfloat y = static_cast<GLfloat>(FLAGS_num_rows - ii);

      // Bottom left, bottom right, top right, top left.
      glVertex2f(x + kEpsilon, y - 1.0 + kEpsilon);
      glVertex2f(x + 1.0 - kEpsilon, y - 1.0 + kEpsilon);
      glVertex2f(x + 1.0 - kEpsilon, y - kEpsilon);
      glVertex2f(x + kEpsilon, y - kEpsilon);
    }
  }
  glEnd();

  // Draw the current state as a circle (polygon with a ton of sides).
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

  // Set up grid world.
  world = new rl::GridWorld(FLAGS_num_rows, FLAGS_num_cols);

  // Create initial and goal states at opposite corners.
  current_state = new rl::GridState(0, 0);
  goal_state = new rl::GridState(FLAGS_num_rows - 1, FLAGS_num_cols - 1);

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
