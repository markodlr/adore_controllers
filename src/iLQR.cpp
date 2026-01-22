/********************************************************************************
 * Copyright (c) 2025 Contributors to the Eclipse Foundation
 *
 * See the NOTICE file(s) distributed with this work for additional
 * information regarding copyright ownership.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License 2.0 which is available at
 * https://www.eclipse.org/legal/epl-2.0
 *
 * SPDX-License-Identifier: EPL-2.0
 ********************************************************************************/

#include "controllers/iLQR.hpp"

#include "dynamics/integration.hpp"

namespace adore
{
namespace controllers
{

void
iLQR::initialize_matrices( int T )
{
  // Initialize dynamics matrices
  A_list = std::vector<Eigen::MatrixXd>( T - 1, Eigen::MatrixXd::Zero( n_x, n_x ) );
  B_list = std::vector<Eigen::MatrixXd>( T - 1, Eigen::MatrixXd::Zero( n_x, n_u ) );

  // Cost derivatives
  l_x_list  = std::vector<Eigen::VectorXd>( T, Eigen::VectorXd::Zero( n_x ) );
  l_u_list  = std::vector<Eigen::VectorXd>( T, Eigen::VectorXd::Zero( n_u ) );
  l_xx_list = std::vector<Eigen::MatrixXd>( T, Eigen::MatrixXd::Zero( n_x, n_x ) );
  l_uu_list = std::vector<Eigen::MatrixXd>( T, Eigen::MatrixXd::Zero( n_u, n_u ) );
  l_xu_list = std::vector<Eigen::MatrixXd>( T, Eigen::MatrixXd::Zero( n_x, n_u ) );

  // Control gains
  K_list = std::vector<Eigen::MatrixXd>( T - 1, Eigen::MatrixXd::Zero( n_u, n_x ) );
  k_list = std::vector<Eigen::VectorXd>( T - 1, Eigen::VectorXd::Zero( n_u ) );
}

double
iLQR::calculate_cost( const dynamics::VehicleStateDynamic& x_ref, const dynamics::VehicleStateDynamic& xt,
                      const dynamics::VehicleCommand& u, const dynamics::VehicleCommand& u_prev, bool is_first_step )
{
  // Compute reference direction (unit vector in x_ref direction)
  double ref_cos = std::cos( x_ref.yaw_angle );
  double ref_sin = std::sin( x_ref.yaw_angle );

  // Decompose position error into longitudinal and lateral components
  double dx                 = xt.x - x_ref.x;
  double dy                 = xt.y - x_ref.y;
  double longitudinal_error = dx * ref_cos + dy * ref_sin;  // projection on reference direction
  double lateral_error      = -dx * ref_sin + dy * ref_cos; // perpendicular to reference direction

  // Velocity and yaw errors
  double dv   = xt.vx - x_ref.vx;
  double dyaw = adore::math::normalize_angle( xt.yaw_angle - x_ref.yaw_angle );

  // Compute jerk and steering rate
  double jerk          = u.acceleration - u_prev.acceleration;
  double steering_rate = u.steering_angle - u_prev.steering_angle;

  // Total cost with separate longitudinal and lateral terms
  double current_cost = longitudinal_weight * longitudinal_error * longitudinal_error + lateral_weight * lateral_error * lateral_error
                      + vel_weight * dv * dv + acc_weight * u.acceleration * u.acceleration
                      + steer_weight * u.steering_angle * u.steering_angle + heading_weight * dyaw * dyaw + jerk_weight * jerk * jerk
                      + steer_rate_weight * steering_rate * steering_rate;

  // Add initial control deviation penalty for first step
  if( is_first_step )
  {
    double accel_deviation = u.acceleration - u_prev.acceleration;
    double steer_deviation = u.steering_angle - u_prev.steering_angle;
    current_cost += initial_control_weight * ( accel_deviation * accel_deviation + steer_deviation * steer_deviation );
  }

  return current_cost * dt;
}

void
iLQR::compute_cost_derivatives( const int T, adore::dynamics::Trajectory& x_traj, const adore::dynamics::Trajectory& ref_trajectory,
                                std::vector<adore::dynamics::VehicleCommand>& u_traj, const dynamics::VehicleCommand& prev_u )
{
  for( int t = 0; t < T; ++t )
  {
    const dynamics::VehicleStateDynamic& xt      = x_traj.states[t];
    const auto&                          x_ref   = ref_trajectory.states[t];
    const dynamics::VehicleCommand&      ut      = ( t < T - 1 ) ? u_traj[t] : dynamics::VehicleCommand( 0, 0 );
    const dynamics::VehicleCommand&      ut_prev = ( t > 0 ) ? u_traj[t - 1] : prev_u;

    // Compute reference direction
    double ref_cos = std::cos( x_ref.yaw_angle );
    double ref_sin = std::sin( x_ref.yaw_angle );

    // Compute longitudinal and lateral errors
    double dx                 = xt.x - x_ref.x;
    double dy                 = xt.y - x_ref.y;
    double longitudinal_error = dx * ref_cos + dy * ref_sin;
    double lateral_error      = -dx * ref_sin + dy * ref_cos;

    // Velocity and yaw errors
    double dv   = xt.vx - x_ref.vx;
    double dyaw = adore::math::normalize_angle( xt.yaw_angle - x_ref.yaw_angle );

    // Compute jerk and steering rate
    double jerk          = ut.acceleration - ut_prev.acceleration;
    double steering_rate = ut.steering_angle - ut_prev.steering_angle;

    // Cost derivatives, scaled by dt
    Eigen::VectorXd l_x = Eigen::VectorXd::Zero( n_x );
    l_x( 0 )            = 2.0 * ( longitudinal_weight * longitudinal_error * ref_cos - lateral_weight * lateral_error * ref_sin ) * dt;
    l_x( 1 )            = 2.0 * ( longitudinal_weight * longitudinal_error * ref_sin + lateral_weight * lateral_error * ref_cos ) * dt;
    l_x( 2 )            = 2.0 * heading_weight * dyaw * dt;
    l_x( 3 )            = 2.0 * vel_weight * dv * dt;

    Eigen::VectorXd l_u = Eigen::VectorXd::Zero( n_u );
    if( t < T - 1 )
    {
      l_u( 0 ) = ( 2.0 * acc_weight * ut.acceleration + 2.0 * jerk_weight * jerk ) * dt;
      l_u( 1 ) = ( 2.0 * steer_weight * ut.steering_angle + 2.0 * steer_rate_weight * steering_rate ) * dt;

      // Add initial control deviation gradient for first step
      if( t == 0 )
      {
        double accel_deviation = ut.acceleration - prev_u.acceleration;
        double steer_deviation = ut.steering_angle - prev_u.steering_angle;
        l_u( 0 ) += 2.0 * initial_control_weight * accel_deviation * dt;
        l_u( 1 ) += 2.0 * initial_control_weight * steer_deviation * dt;
      }
    }

    // Hessians, scaled by dt
    Eigen::MatrixXd l_xx = Eigen::MatrixXd::Zero( n_x, n_x );
    l_xx( 0, 0 )         = ( 2.0 * longitudinal_weight * ref_cos * ref_cos + 2.0 * lateral_weight * ref_sin * ref_sin ) * dt;
    l_xx( 1, 1 )         = ( 2.0 * longitudinal_weight * ref_sin * ref_sin + 2.0 * lateral_weight * ref_cos * ref_cos ) * dt;
    l_xx( 2, 2 )         = 2.0 * heading_weight * dt;
    l_xx( 3, 3 )         = 2.0 * vel_weight * dt;

    Eigen::MatrixXd l_uu = Eigen::MatrixXd::Zero( n_u, n_u );
    if( t < T - 1 )
    {
      l_uu( 0, 0 ) = ( 2.0 * acc_weight + 2.0 * jerk_weight ) * dt;
      l_uu( 1, 1 ) = ( 2.0 * steer_weight + 2.0 * steer_rate_weight ) * dt;

      // Add initial control deviation Hessian for first step
      if( t == 0 )
      {
        l_uu( 0, 0 ) += 2.0 * initial_control_weight * dt;
        l_uu( 1, 1 ) += 2.0 * initial_control_weight * dt;
      }
    }

    l_x_list[t]  = l_x;
    l_u_list[t]  = l_u;
    l_xx_list[t] = l_xx;
    l_uu_list[t] = l_uu;
  }
}

void
iLQR::set_parameters( const std::map<std::string, double>& params )
{
  for( const auto& [name, value] : params )
  {
    if( name == "horizon_steps" )
      horizon_steps = static_cast<int>( value );
    else if( name == "max_iterations" )
      max_iterations = static_cast<int>( value );
    else if( name == "dt" )
      dt = value;
    else if( name == "heading_weight" )
      heading_weight = value;
    else if( name == "vel_weight" )
      vel_weight = value;
    else if( name == "acc_weight" )
      acc_weight = value;
    else if( name == "steer_weight" )
      steer_weight = value;
    else if( name == "jerk_weight" )
      jerk_weight = value;
    else if( name == "steer_rate_weight" )
      steer_rate_weight = value;
    else if( name == "convergence_threshold" )
      convergence_threshold = value;
    else if( name == "longitudinal_weight" )
      longitudinal_weight = value; // Set longitudinal weight
    else if( name == "lateral_weight" )
      lateral_weight = value; // Set lateral weight
    else if( name == "initial_control_weight" )
      initial_control_weight = value;
    else if( name == "debug active" )
      debug_active = static_cast<bool>( value );

    std::cerr << name << " : " << value << std::endl;
  }
}

dynamics::VehicleCommand
iLQR::get_next_vehicle_command( const dynamics::Trajectory& in_trajectory, const dynamics::VehicleStateDynamic& current_state )
{
  const size_t                          T = std::min( horizon_steps, in_trajectory.states.size() );
  std::vector<dynamics::VehicleCommand> u_traj( T, dynamics::VehicleCommand( 0.0, 0.0 ) );

  dynamics::Trajectory ref_trajectory; // = in_trajectory;
  double               start_time = current_state.time;
  for( size_t i = 0; i < T; i++ )
  {
    ref_trajectory.states.push_back( in_trajectory.get_state_at_time( start_time + i * dt ) );
  }

  // Start timing
  auto start_timer = std::chrono::high_resolution_clock::now();

  warm_start( u_traj );
  initialize_matrices( T );

  dynamics::Trajectory x_traj;
  x_traj.states.resize( T );

  double last_total_cost     = std::numeric_limits<double>::infinity();
  bool   convergence_reached = false;
  size_t iteration           = 0;
  while( iteration < max_iterations )
  {
    iteration++;
    x_traj.states[0] = current_state;
    for( size_t t = 0; t < T - 1; ++t )
    {
      x_traj.states[t + 1] = dynamics::integrate_euler( x_traj.states[t], u_traj[t], dt, model.motion_model );
    }

    // Calculate total cost for the current iteration
    double total_cost = 0.0;
    for( size_t t = 0; t < T - 1; ++t )
    {
      const dynamics::VehicleCommand& u_prev = ( t > 0 ) ? u_traj[t - 1]
                                                         : dynamics::VehicleCommand( current_state.steering_angle, current_state.ax );
      total_cost += calculate_cost( ref_trajectory.states[t], x_traj.states[t], u_traj[t], u_prev, t == 0 );
    }

    // Check convergence
    if( std::abs( last_total_cost - total_cost ) < convergence_threshold )
    {
      convergence_reached = true;
      last_total_cost     = total_cost;
      break;
    }

    last_total_cost = total_cost;

    // Perform linearization and update derivatives
    extract_dynamics_linearization( T, x_traj, u_traj );
    compute_cost_derivatives( T, x_traj, ref_trajectory, u_traj,
                              dynamics::VehicleCommand( current_state.steering_angle, current_state.ax ) );
    backward_pass( T );

    double       line_step = 1.0;
    const double min_step  = 1e-5;
    bool         success   = line_search( line_step, min_step, T, current_state, x_traj, u_traj, ref_trajectory, last_total_cost );


    if( !success )
      break;
  }

  // Calculate time taken
  auto                          end_timer       = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end_timer - start_timer;

  if( debug_active )
  {
    // Log cost, time taken, and convergence status
    std::cerr << "iLQR total cost: " << last_total_cost << "\n";
    std::cerr << "iLQR execution time: " << elapsed_seconds.count() << " seconds\n";
    std::cerr << "iLQR iterations: " << iteration << " \n";
    std::cerr << "iLQR dt: " << dt << "\n";
    std::cerr << "Convergence criteria met: " << ( convergence_reached ? "Yes" : "No" ) << "\n";
  }


  // Prepare and return the next command
  dynamics::VehicleCommand command;
  command.acceleration   = ( u_traj[0].acceleration + u_traj[1].acceleration ) / 2;
  command.steering_angle = ( u_traj[0].steering_angle + u_traj[1].steering_angle ) / 2;

  // Save the optimized trajectory for warm start
  previous_u_traj = u_traj;
  for( size_t i = 0; i < u_traj.size(); i++ )
  {
    x_traj.states[i].steering_angle = u_traj[i].steering_angle;
    x_traj.states[i].ax             = u_traj[i].acceleration;
  }
  previous_traj = x_traj;


  return command;
}

void
iLQR::warm_start( std::vector<adore::dynamics::VehicleCommand>& u_traj )
{
  size_t max_T = u_traj.size() < previous_u_traj.size() ? u_traj.size() : previous_u_traj.size();
  // Shift the previous trajectory one step forward
  for( size_t t = 0; t < max_T; ++t )
  {
    u_traj[t] = previous_u_traj[t + 1];
  }
}

bool
iLQR::line_search( double& line_step, const double min_step, const int T, const adore::dynamics::VehicleStateDynamic& current_state,
                   adore::dynamics::Trajectory& x_traj, std::vector<adore::dynamics::VehicleCommand>& u_traj,
                   const adore::dynamics::Trajectory& ref_trajectory, double last_total_cost )
{
  const double alpha = 0.2; // Armijo condition parameter (typically between 0.1 and 0.3)
  const double beta  = 0.5; // Reduction factor for line step (typical range is 0.5 to 0.8)

  while( line_step > min_step )
  {
    std::vector<dynamics::VehicleCommand> u_traj_new( T - 1 );
    dynamics::Trajectory                  x_traj_new;
    x_traj_new.states.resize( T );
    x_traj_new.states[0] = current_state;

    double predicted_decrease = 0.0;

    for( int t = 0; t < T - 1; ++t )
    {
      // Control update
      Eigen::VectorXd dx = Eigen::VectorXd( n_x );
      dx( 0 )            = x_traj_new.states[t].x - x_traj.states[t].x;
      dx( 1 )            = x_traj_new.states[t].y - x_traj.states[t].y;
      dx( 2 )            = x_traj_new.states[t].yaw_angle - x_traj.states[t].yaw_angle;
      dx( 3 )            = x_traj_new.states[t].vx - x_traj.states[t].vx;

      Eigen::VectorXd          du    = k_list[t] + K_list[t] * dx;
      dynamics::VehicleCommand u_new = u_traj[t];
      u_new.acceleration += line_step * du[0];
      u_new.steering_angle += line_step * du[1];

      // Apply control limits if necessary
      u_new.clamp_within_limits( model.params );
      u_traj_new[t] = u_new;

      // Simulate dynamics
      x_traj_new.states[t + 1] = dynamics::integrate_euler( x_traj_new.states[t], u_new, dt, model.motion_model );

      // Calculate predicted decrease as -grad(J) * du * line_step
      Eigen::VectorXd grad_J = l_u_list[t]; // Gradient of cost wrt control at time t
      predicted_decrease += grad_J.dot( du ) * line_step;
    }

    // Compute new total cost
    double new_total_cost = 0.0;
    for( int t = 0; t < T - 1; ++t )
    {
      const dynamics::VehicleCommand& u_prev = ( t > 0 ) ? u_traj_new[t - 1]
                                                         : dynamics::VehicleCommand( current_state.steering_angle, current_state.ax );
      new_total_cost += calculate_cost( ref_trajectory.states[t], x_traj_new.states[t], u_traj_new[t], u_prev, t == 0 );
    }

    // Armijo condition: Check if the reduction is sufficient
    if( new_total_cost <= last_total_cost + alpha * line_step * predicted_decrease )
    {
      // Accept the step size
      u_traj = u_traj_new;
      x_traj = x_traj_new;
      return true;
    }
    else
    {
      // Reduce line step
      line_step *= beta;
    }
  }
  return false; // Return false if no valid line step is found
}

void
iLQR::backward_pass( const int T )
{
  Eigen::VectorXd V_x  = l_x_list.back();
  Eigen::MatrixXd V_xx = l_xx_list.back();

  // Variable to hold the previous Q_uu factorization
  static Eigen::MatrixXd             prev_Q_uu;
  static bool                        factor_cached = false;
  static Eigen::LLT<Eigen::MatrixXd> cached_cholesky;

  for( int t = T - 2; t >= 0; --t )
  {
    const Eigen::MatrixXd& A    = A_list[t];
    const Eigen::MatrixXd& B    = B_list[t];
    const Eigen::VectorXd& l_x  = l_x_list[t];
    const Eigen::VectorXd& l_u  = l_u_list[t];
    const Eigen::MatrixXd& l_xx = l_xx_list[t];
    const Eigen::MatrixXd& l_uu = l_uu_list[t];

    // Q-function derivatives
    Eigen::VectorXd Q_x  = l_x + A.transpose() * V_x;
    Eigen::VectorXd Q_u  = l_u + B.transpose() * V_x;
    Eigen::MatrixXd Q_xx = l_xx + A.transpose() * V_xx * A;
    Eigen::MatrixXd Q_ux = B.transpose() * V_xx * A;
    Eigen::MatrixXd Q_uu = l_uu + B.transpose() * V_xx * B;

    // Regularize Q_uu if needed to ensure positive definiteness
    Eigen::MatrixXd Q_uu_reg = Q_uu;

    // Check if Q_uu has changed significantly from the previous iteration
    if( !factor_cached || ( Q_uu - prev_Q_uu ).norm() > 1e-5 )
    {
      // Update previous Q_uu and cache new factorization
      prev_Q_uu       = Q_uu;
      cached_cholesky = Q_uu_reg.llt(); // Compute the Cholesky factorization
      factor_cached   = true;

      // Ensure Q_uu_reg is positive definite by adding small regularization if necessary
      while( cached_cholesky.info() != Eigen::Success )
      {
        Q_uu_reg += Eigen::MatrixXd::Identity( n_u, n_u ) * 1e-6;
        cached_cholesky = Q_uu_reg.llt();
      }
    }

    // Use cached or freshly computed factorization to get Q_uu_inv
    Eigen::MatrixXd Q_uu_inv = cached_cholesky.solve( Eigen::MatrixXd::Identity( n_u, n_u ) );

    // Compute control gains
    Eigen::VectorXd k = -Q_uu_inv * Q_u;
    Eigen::MatrixXd K = -Q_uu_inv * Q_ux;

    k_list[t] = k;
    K_list[t] = K;

    // Update value function
    V_x  = Q_x + K.transpose() * Q_uu * k + K.transpose() * Q_u + Q_ux.transpose() * k;
    V_xx = Q_xx + K.transpose() * Q_uu * K + K.transpose() * Q_ux + Q_ux.transpose() * K;

    // Ensure symmetry of V_xx
    V_xx = 0.5 * ( V_xx + V_xx.transpose() );
  }
}

void
iLQR::extract_dynamics_linearization( const int T, adore::dynamics::Trajectory& x_traj,
                                      std::vector<adore::dynamics::VehicleCommand>& u_traj )
{
  for( int t = 0; t < T - 1; ++t )
  {
    const dynamics::VehicleStateDynamic& xt = x_traj.states[t];

    const double& yaw   = xt.yaw_angle;
    const double& vx    = xt.vx;
    const double& delta = u_traj[t].steering_angle;

    double cos_yaw           = std::cos( yaw );
    double sin_yaw           = std::sin( yaw );
    double tan_delta         = std::tan( delta );
    double sec_delta         = 1.0 / std::cos( delta );
    double sec_delta_squared = sec_delta * sec_delta;

    // State derivatives
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero( n_x, n_x );
    A( 0, 0 )         = 1.0;
    A( 0, 2 )         = -vx * sin_yaw * dt;
    A( 0, 3 )         = cos_yaw * dt;
    A( 1, 1 )         = 1.0;
    A( 1, 2 )         = vx * cos_yaw * dt;
    A( 1, 3 )         = sin_yaw * dt;
    A( 2, 2 )         = 1.0;
    A( 2, 3 )         = ( tan_delta / model.params.wheelbase ) * dt;
    A( 3, 3 )         = 1.0;

    // Control derivatives
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero( n_x, n_u );
    B( 2, 1 )         = ( vx / model.params.wheelbase ) * sec_delta_squared * dt;
    B( 3, 0 )         = dt;

    A_list[t] = A;
    B_list[t] = B;
  }
}

dynamics::Trajectory
controllers::iLQR::get_last_trajectory() const // for visualizing
{
  return previous_traj;
}
} // namespace controllers


} // namespace adore
