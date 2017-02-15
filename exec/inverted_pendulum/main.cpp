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

#include <environment/inverted_pendulum.hpp>

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

// Animation parameters.
DEFINE_int32(refresh_rate, 10, "Refresh rate in milliseconds.");
DEFINE_double(motion_rate, 0.5, "Fraction of real-time.");

// Solver parameters.
DEFINE_int32(num_value_updates, 1,
             "Number of value updates per policy iteration.");
DEFINE_int32(max_iterations, 100,
             "Maximum umber of iterations to run modified policy iteration.");
DEFINE_double(discount_factor, 0.9, "Discount factor.");

// Environment parameters.
DEFINE_double(arm_length, 1.0, "Length of pendulum arm in meters.");
DEFINE_double(ball_radius, 0.1, "Ball radius in meters.");
DEFINE_double(ball_mass, 1.0, "Ball mass in kilograms.");
DEFINE_double(initial_theta, 0.25 * M_PI, "Initial angle from the +x axis.");
DEFINE_double(initial_omega, 0.0, "Initial angular velocity.");
DEFINE_double(friction, 0.01, "Torque applied by friction.");
DEFINE_double(torque_limit, 0.1, "Limit for applied torque.");
DEFINE_double(time_step, 0.001, "Time step for numerical integration.");

// Create a globally-defined simulator and current state.
InvertedPendulum* world = NULL;
InvertedPendulumState* current_state = NULL;

// Create a solver.
//ModifiedPolicyIteration<GridState, GridAction>* solver = NULL;

// Flag for whether we have reached a terminal state.
bool is_terminal = false;

// Initialize OpenGL.
void InitGL() {
  // Set the "clearing" or background color as black/opaque.
  glClearColor(0.0, 0.0, 0.0, 1.0);

  // Set up alpha blending.
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_BLEND);
}

// Visualize.
void Visualize() {
  // Visualize environment.
  world->Visualize();

  // Draw the current state.
  current_state->Visualize(FLAGS_arm_length, FLAGS_ball_radius);

  // Swap buffers.
  glutSwapBuffers();
}

// Timer callback. Re-render at the specified rate if not terninal.
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

  // Compute aspect ratio of the new window and for the pendulum.
  const GLfloat kWindowRatio =
    static_cast<GLfloat>(width) / static_cast<GLfloat>(height);

  const GLfloat kHorizontalExtent = 1.5 * static_cast<GLfloat>(FLAGS_arm_length);
  const GLfloat kBottomExtent = 0.1 * static_cast<GLfloat>(FLAGS_arm_length);
  const GLfloat kTopExtent = 1.5 * static_cast<GLfloat>(FLAGS_arm_length);

  const GLfloat kPendulumRatio =
    2.0 * kHorizontalExtent / (kTopExtent + kBottomExtent);

  // Set the viewport to cover the new window.
  glViewport(0, 0, width, height);

  // Set the clipping area.
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  // One of two possibilities:
  if (kPendulumRatio >= kWindowRatio) {
    // (1) kPendulumRatio >= kWindowRatio, in which case horizontal dimensions
    //     match but vertical dimensions must be scaled up.
    const GLfloat kVerticalScaling = kPendulumRatio / kWindowRatio;
    gluOrtho2D(-kHorizontalExtent, kHorizontalExtent,
               -kBottomExtent * kVerticalScaling, kTopExtent * kVerticalScaling);
  } else {
    // (2) kPendulumRatio < kWindowRatio, in which case the reverse is true.
    const GLfloat kHorizontalScaling = kWindowRatio / kPendulumRatio;
    gluOrtho2D(-kHorizontalExtent * kHorizontalScaling,
               kHorizontalExtent * kHorizontalScaling,
               -kBottomExtent, kTopExtent);
  }
}

// Take a single step.
void SingleIteration() {
  CHECK_NOTNULL(world);
  CHECK_NOTNULL(current_state);
  //  CHECK_NOTNULL(solver);

  // Extract optimal policy and value function.
  //  const DiscreteDeterministicPolicy<GridState, GridAction> policy =
  //    solver->Policy();
  //  const DiscreteStateValueFunctor<GridState> value = solver->Value();

  // Check for terminal state.
  if (world->IsTerminal(*current_state))
    is_terminal = true;
  else {
    // If not a terminal state, update state.
    // Get optimal action.
    //    GridAction action;
    //    policy.Act(*current_state, action);

    // Simulate this action.
    const double reward = world->Simulate(*current_state, 0.0);

    // Print out step counter and reward.
    std::printf("Received reward of %f.\n", reward);
  }

  // Visualize.
  Visualize();
}

// Set everything up and go!
int main(int argc, char** argv) {
  // Set up logging.
  google::InitGoogleLogging(argv[0]);

  // Parse flags.
  google::ParseCommandLineFlags(&argc, &argv, true);

  // Create initial state.
  current_state =
    new InvertedPendulumState(FLAGS_initial_theta, FLAGS_initial_omega);

  // Set up grid world.
  InvertedPendulumParams params;
  params.arm_length_ = FLAGS_arm_length;
  params.ball_radius_ = FLAGS_ball_radius;
  params.ball_mass_ = FLAGS_ball_mass;
  params.friction_ = FLAGS_friction;
  params.torque_lower_ = -FLAGS_torque_limit;
  params.torque_upper_ = FLAGS_torque_limit;
  params.time_step_ = FLAGS_time_step;
  params.control_period_ =
    0.001 * static_cast<double>(FLAGS_refresh_rate) * FLAGS_motion_rate;
  world = new InvertedPendulum(params);

  // Set up the solver.
  //  solver = new ModifiedPolicyIteration<GridState, GridAction>(
  //     FLAGS_num_value_updates, FLAGS_max_iterations, FLAGS_discount_factor);

  //  if (solver->Solve(*world)) {
    // Set up OpenGL window.
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE);
    glutInitWindowSize(320, 320);
    glutInitWindowPosition(50, 50);
    glutCreateWindow("Inverted Pendulum");
    glutDisplayFunc(SingleIteration);
    glutReshapeFunc(Reshape);
    glutTimerFunc(0, Timer, 0);
    InitGL();
    glutMainLoop();

    delete world;
    delete current_state;
    return 0;
    //  }
#if 0
  std::printf("Solver did not converge.\n");

  delete world;
  delete current_state;
  delete goal_state;
  return 1;
#endif
}
