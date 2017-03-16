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
#include <value/deep_action_value_functor.hpp>
#include <value/gaussian_action_value_functor.hpp>
#include <value/linear_action_value_functor.hpp>
#include <solver/continuous_q_learning.hpp>

#include <mininet/loss/l2.h>

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
using namespace mininet;

// Animation parameters.
DEFINE_int32(refresh_rate, 10, "Refresh rate in milliseconds.");
DEFINE_double(motion_rate, 0.25, "Fraction of real-time.");
DEFINE_int32(replan_rate, 1000, "Replanning rate in milliseconds.");

// Value function approximator.
DEFINE_string(value_approx, "gp",
              "Type of value function approx to use of {linear, deep, gp}.");

// GP value functor params.
DEFINE_int32(gp_num_points, 50, "Number of random training points for GP.");
DEFINE_int32(gp_num_inits, 10, "Number of random initial actions.");
DEFINE_int32(gp_max_steps, 10,
             "Maximum number of gradient steps to find the optimal action.");
DEFINE_double(gp_step_size, 0.01, "Step size for finding optimal action.");
DEFINE_double(gp_epsilon, 0.1,
              "Gradient size threshold for convergence of optimal action.");
DEFINE_double(gp_regularizer, 1.0,
              "Regularization constant to trade off mean/variance.");
DEFINE_double(gp_noise, 0.1, "Noise variance of GP.");
DEFINE_double(gp_state_length, 3.0, "Length scale for state dimensions.");
DEFINE_double(gp_action_length, 10.0, "Length scale for action dimensions");

// Deep value functor params.
DEFINE_int32(num_layers, 3, "Number of layers in deep value approximator.");
DEFINE_int32(layer_size, 10, "Size of hidden layers.");
DEFINE_double(momentum, 0.0, "Momentum coefficient in SGD.");
DEFINE_double(weight_decay, 0.0, "L2 loss coefficient for weights.");
DEFINE_string(nonlinearity, "relu",
              "Nonlinearity on all but the last layer {relu, sigmoid}.");

// Solver parameters.
DEFINE_double(discount_factor, 0.9, "Discount factor.");
DEFINE_double(alpha, 0.01, "TD return interpolation parameter.");
DEFINE_double(learning_rate, 1.0, "Learning rate for SGD.");
DEFINE_double(learning_rate_decay, 0.9, "Learning rate decay.");
DEFINE_int32(num_rollouts, 100, "Number of rollouts to learn from.");
DEFINE_int32(rollout_length, 100,
             "Rollout length. If negative, rollout until a terminal state.");
DEFINE_int32(num_exp_replays, 1000,
             "Number of SGD updates per Q Learning iteration.");
DEFINE_int32(batch_size, 5,
             "Number of experience replays per SGD iteration.");

// Environment parameters.
DEFINE_double(arm_length, 1.0, "Length of pendulum arm in meters.");
DEFINE_double(ball_radius, 0.1, "Ball radius in meters.");
DEFINE_double(ball_mass, 1.0, "Ball mass in kilograms.");
DEFINE_double(initial_theta, 1.25, "Initial angle from the +x axis.");
DEFINE_double(initial_omega, 0.0, "Initial angular velocity.");
DEFINE_double(friction, 5.0, "Torque applied by friction.");
DEFINE_double(torque_limit, 20.0, "Limit for applied torque.");
DEFINE_double(time_step, 0.01, "Time step for numerical integration.");
DEFINE_int32(num_action_values, 5, "Number of discrete action values.");

// Create a globally-defined simulator and current state.
InvertedPendulum* world = NULL;
InvertedPendulumState* current_state = NULL;

// Create a continuous value function.
ContinuousActionValue<InvertedPendulumState, InvertedPendulumAction>::Ptr value;

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

// Replan.
void Replan() {
  CHECK_NOTNULL(current_state);
  CHECK_NOTNULL(world);
  CHECK_NOTNULL(value.get());

  // Set up the solver.
  SolverParams solver_params;
  solver_params.discount_factor_ = FLAGS_discount_factor;
  solver_params.alpha_ = FLAGS_alpha;
  solver_params.num_rollouts_ = FLAGS_num_rollouts;
  solver_params.rollout_length_ = FLAGS_rollout_length;
  solver_params.num_exp_replays_ = FLAGS_num_exp_replays;
  solver_params.batch_size_ = FLAGS_batch_size;
  solver_params.learning_rate_ = FLAGS_learning_rate;
  solver_params.learning_rate_decay_ = FLAGS_learning_rate_decay;
  ContinuousQLearning<InvertedPendulumState,
                      InvertedPendulumAction> solver(*current_state, solver_params);

  std::cout << "Running solver..." << std::flush;
  solver.Solve(*world, value, true, true);
  std::cout << "done." << std::endl;
}

// Animation timer callback. Re-render at the specified rate.
void AnimationTimer(int value) {
  glutPostRedisplay();
  glutTimerFunc(FLAGS_refresh_rate, AnimationTimer, 0);
}

// Replanning timer callback. Replan every time this fires.
void ReplanningTimer(int value) {
  Replan();
  glutTimerFunc(FLAGS_replan_rate, ReplanningTimer, 0);
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
  CHECK_NOTNULL(value.get());

  // Check for terminal state.
  if (world->IsTerminal(*current_state))
    *current_state = InvertedPendulumState();
  else {
    // If not a terminal state, update state.
    // Extract optimal action from the value function.
    InvertedPendulumAction action;
    if (!value->OptimalAction(*current_state, action)) {
      LOG(WARNING) << "Taking a sub-optimal action.";
    }

    // Simulate this action.
    const double reward = world->Simulate(*current_state, action);

    // Print out step counter and reward.
    std::printf("Received reward of %f for torque of %f.\n",
                reward, action.torque_);
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
  InvertedPendulumParams world_params;
  world_params.arm_length_ = FLAGS_arm_length;
  world_params.ball_radius_ = FLAGS_ball_radius;
  world_params.ball_mass_ = FLAGS_ball_mass;
  world_params.friction_ = FLAGS_friction;
  world_params.torque_lower_ = -FLAGS_torque_limit;
  world_params.torque_upper_ = FLAGS_torque_limit;
  world_params.time_step_ = FLAGS_time_step;
  world_params.control_period_ =
    0.001 * static_cast<double>(FLAGS_refresh_rate) * FLAGS_motion_rate;
  world_params.num_action_values_ = FLAGS_num_action_values;
  world = new InvertedPendulum(world_params);

  // Initialize the value function.
  if (FLAGS_value_approx == "linear")
    value = LinearActionValue<InvertedPendulumState,
                              InvertedPendulumAction>::Create();
  else if (FLAGS_value_approx == "gp") {
    VectorXd lengths =
      VectorXd::Zero(InvertedPendulumState::FeatureDimension() +
                     InvertedPendulumAction::FeatureDimension());
    lengths.head(InvertedPendulumState::FeatureDimension()) =
      VectorXd::Constant(InvertedPendulumState::FeatureDimension(),
                         FLAGS_gp_state_length);
    lengths.tail(InvertedPendulumAction::FeatureDimension()) =
      VectorXd::Constant(InvertedPendulumAction::FeatureDimension(),
                         FLAGS_gp_action_length);

    GaussianParams params;
    params.num_points_ = FLAGS_gp_num_points;
    params.noise_variance_ = FLAGS_gp_noise;
    params.regularizer_ = FLAGS_gp_regularizer;
    params.step_size_ = FLAGS_gp_step_size;
    params.max_steps_ = FLAGS_gp_max_steps;
    params.num_inits_ = FLAGS_gp_num_inits;
    params.epsilon_ = FLAGS_gp_epsilon;
    params.lengths_ = lengths;

    value = GaussianActionValue<InvertedPendulumState,
                                       InvertedPendulumAction>::Create(params);
  } else if (FLAGS_value_approx == "deep") {
    CHECK_GE(FLAGS_num_layers, 1);

    // Parse layer type.
    LayerType nonlinearity = SIGMOID;
    if (FLAGS_nonlinearity == "relu")
      nonlinearity = RELU;
    else if (FLAGS_nonlinearity == "linear")
      nonlinearity = LINEAR;

    // Create layers.
    std::vector<LayerParams> layers;

    if (FLAGS_num_layers == 1)
      // Handle single layer.
      layers.push_back(LayerParams(LINEAR,
                                   InvertedPendulumState::FeatureDimension() +
                                   InvertedPendulumAction::FeatureDimension(),
                                   InvertedPendulumAction::FeatureDimension()));
    else {
      // Handle multiple layers.
      layers.push_back(LayerParams(nonlinearity,
                                   InvertedPendulumState::FeatureDimension() +
                                   InvertedPendulumAction::FeatureDimension(),
                                   FLAGS_layer_size));

      for (size_t ii = 1; ii < FLAGS_num_layers - 1; ii++)
        layers.push_back(LayerParams(nonlinearity,
                                     FLAGS_layer_size, FLAGS_layer_size));

      layers.push_back(LayerParams(LINEAR, FLAGS_layer_size,
                                   InvertedPendulumAction::FeatureDimension()));
    }

    // Create L2 loss functor.
    LossFunctor::ConstPtr loss = L2::Create();

    value = DeepActionValue<
      InvertedPendulumState, InvertedPendulumAction>::Create(
      layers, loss, FLAGS_momentum, FLAGS_weight_decay);
  }

  // Set up the solver.
  Replan();

  // Set up OpenGL window.
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE);
  glutInitWindowSize(640, 640);
  glutInitWindowPosition(50, 50);
  glutCreateWindow("Inverted Pendulum");
  glutDisplayFunc(SingleIteration);
  glutReshapeFunc(Reshape);
  glutTimerFunc(0, AnimationTimer, 0);
  //  glutTimerFunc(0, ReplanningTimer, 0);
  InitGL();
  glutMainLoop();

  delete world;
  delete current_state;
  return 0;
}
