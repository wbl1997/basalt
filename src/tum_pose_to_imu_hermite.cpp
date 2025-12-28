/**
 * TUM Pose to IMU Data Generator
 *
 * Use direct numerical differentiation on poses to avoid B-spline phase lag.
 * This approach guarantees zero phase error between pose and IMU.
 *
 * Usage:
 *   ./tum_pose_to_imu <input_tum_file> <output_imu_file> [freq]
 */

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <rosbag/bag.h>
#include <sensor_msgs/Imu.h>

static const Eigen::Vector3d g(0, 0, -9.81);

static const double GYRO_NOISE_DENSITY     = 1.6968e-04;
static const double ACCEL_NOISE_DENSITY    = 2.0000e-3;
static const double GYRO_BIAS_RANDOM_WALK  = 1.9393e-05;
static const double ACCEL_BIAS_RANDOM_WALK = 3.0000e-03;

struct PoseData {
  double t;
  Eigen::Vector3d pos;
  Eigen::Quaterniond q;
};

static void load_tum_trajectory(const std::string& path, std::vector<PoseData>& poses) {
  std::ifstream is(path);
  if (!is.is_open()) {
    throw std::runtime_error("Failed to open: " + path);
  }

  std::string line;
  Eigen::Quaterniond q_prev;
  bool has_prev = false;

  while (std::getline(is, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::stringstream ss(line);
    double t_s, tx, ty, tz, qx, qy, qz, qw;
    if (!(ss >> t_s >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) {
      continue;
    }

    Eigen::Quaterniond q(qw, qx, qy, qz);
    q.normalize();

    if (has_prev && q_prev.dot(q) < 0.0) {
      q.coeffs() *= -1.0;
    }
    q_prev = q;
    has_prev = true;

    PoseData pd;
    pd.t = t_s;
    pd.pos = Eigen::Vector3d(tx, ty, tz);
    pd.q = q;
    poses.push_back(pd);
  }

  if (poses.size() < 2) {
    throw std::runtime_error("Trajectory has < 2 poses.");
  }

  // Sort by time
  std::sort(poses.begin(), poses.end(), [](const PoseData& a, const PoseData& b) {
    return a.t < b.t;
  });

  // Remove duplicates
  std::vector<PoseData> clean;
  clean.push_back(poses[0]);
  for (size_t i = 1; i < poses.size(); ++i) {
    if (poses[i].t > clean.back().t + 1e-9) {
      clean.push_back(poses[i]);
    }
  }
  poses.swap(clean);

  std::cout << "Loaded " << poses.size() << " poses from " << path << std::endl;
}

// Cubic Hermite interpolation for acceleration (second derivative)
static Eigen::Vector3d hermite_interp_accel(
    double t, double t0, double t1,
    const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
    const Eigen::Vector3d& v0, const Eigen::Vector3d& v1) {
  double dt = t1 - t0;
  double s = (t - t0) / dt;
  
  double ddh00 = (12*s - 6) / (dt * dt);
  double ddh10 = (6*s - 4) / dt;
  double ddh01 = (-12*s + 6) / (dt * dt);
  double ddh11 = (6*s - 2) / dt;
  
  return ddh00 * p0 + ddh10 * v0 + ddh01 * p1 + ddh11 * v1;
}

// SLERP for quaternion interpolation
static Eigen::Quaterniond slerp(const Eigen::Quaterniond& q0, const Eigen::Quaterniond& q1, double t) {
  return q0.slerp(t, q1);
}

// Compute angular velocity from quaternion derivative
// omega_body = 2 * q_conj * dq/dt
static Eigen::Vector3d compute_angular_velocity(
    const Eigen::Quaterniond& q, const Eigen::Quaterniond& dq_dt) {
  Eigen::Quaterniond omega_quat = q.conjugate() * dq_dt;
  return 2.0 * Eigen::Vector3d(omega_quat.x(), omega_quat.y(), omega_quat.z());
}

static bool ends_with(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

int main(int argc, char** argv) {
  try {
    if (argc < 3) {
      std::cerr << "Usage: " << argv[0]
                << " <input_tum_file> <output_imu_file> [freq]\n";
      return 1;
    }

    const std::string input_path  = argv[1];
    const std::string output_path = argv[2];
    const int imu_freq = (argc > 3) ? std::stoi(argv[3]) : 200;
    const double dt_imu = 1.0 / imu_freq;

    // 1) Load trajectory
    std::vector<PoseData> poses;
    load_tum_trajectory(input_path, poses);

    const double t_start = poses.front().t;
    const double t_end = poses.back().t;
    const double duration = t_end - t_start;
    
    std::cout << "Time range: [" << t_start << ", " << t_end << "] s (duration: " << duration << " s)\n";
    std::cout << "IMU freq: " << imu_freq << " Hz\n";

    // 2) Compute velocities at each pose using central differences
    std::vector<Eigen::Vector3d> velocities(poses.size());
    
    // First point: forward difference
    velocities[0] = (poses[1].pos - poses[0].pos) / (poses[1].t - poses[0].t);
    
    // Middle points: central difference
    for (size_t i = 1; i < poses.size() - 1; ++i) {
      double dt_prev = poses[i].t - poses[i-1].t;
      double dt_next = poses[i+1].t - poses[i].t;
      double dt_total = poses[i+1].t - poses[i-1].t;
      
      // Weighted central difference for non-uniform spacing
      Eigen::Vector3d v_prev = (poses[i].pos - poses[i-1].pos) / dt_prev;
      Eigen::Vector3d v_next = (poses[i+1].pos - poses[i].pos) / dt_next;
      velocities[i] = (v_prev * dt_next + v_next * dt_prev) / dt_total;
    }
    
    // Last point: backward difference
    size_t n = poses.size();
    velocities[n-1] = (poses[n-1].pos - poses[n-2].pos) / (poses[n-1].t - poses[n-2].t);

    // 3) Output setup
    const bool output_to_bag = ends_with(output_path, ".bag");
    std::ofstream os;
    std::unique_ptr<rosbag::Bag> bag;

    if (output_to_bag) {
      bag.reset(new rosbag::Bag);
      bag->open(output_path, rosbag::bagmode::Write);
      std::cout << "Output: ROS bag " << output_path << " (topic /simu/imu)\n";
    } else {
      os.open(output_path);
      if (!os.is_open()) throw std::runtime_error("Failed to open output: " + output_path);
      os << std::fixed << std::setprecision(9);
      os << "# timestamp(s) wx wy wz ax ay az qx qy qz qw\n";
      std::cout << "Output: text " << output_path << "\n";
    }

    // 4) Noise setup
    std::mt19937 gen{1};
    const double gyro_noise_std  = GYRO_NOISE_DENSITY / std::sqrt(dt_imu);
    const double accel_noise_std = ACCEL_NOISE_DENSITY / std::sqrt(dt_imu);

    std::normal_distribution<> gyro_noise_dist{0.0, gyro_noise_std};
    std::normal_distribution<> accel_noise_dist{0.0, accel_noise_std};
    std::normal_distribution<> gyro_bias_dist{0.0, GYRO_BIAS_RANDOM_WALK};
    std::normal_distribution<> accel_bias_dist{0.0, ACCEL_BIAS_RANDOM_WALK};

    const bool enable_noise = true;
    Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro_bias  = Eigen::Vector3d::Zero();
    if (enable_noise) {
        accel_bias = Eigen::Vector3d::Random() / 10.0;
        gyro_bias  = Eigen::Vector3d::Random() / 100.0;
    }

    // 5) Sample IMU using Hermite interpolation
    std::cout << "\n=== Generating IMU data ===\n";
    int sample_count = 0;
    size_t seg_idx = 0;  // Current segment index

    // Small epsilon for numerical differentiation of quaternion
    const double eps = 1e-6;

    for (double t = t_start; t < t_end - eps; t += dt_imu) {
      // Find the segment containing t
      while (seg_idx < poses.size() - 2 && poses[seg_idx + 1].t <= t) {
        ++seg_idx;
      }
      
      if (seg_idx >= poses.size() - 1) break;

      const PoseData& p0 = poses[seg_idx];
      const PoseData& p1 = poses[seg_idx + 1];
      
      // Interpolate acceleration using Hermite spline
      Eigen::Vector3d acc_world = hermite_interp_accel(t, p0.t, p1.t, p0.pos, p1.pos,
                                                        velocities[seg_idx], velocities[seg_idx + 1]);
      
      // Interpolate quaternion using SLERP
      double alpha = (t - p0.t) / (p1.t - p0.t);
      Eigen::Quaterniond q = slerp(p0.q, p1.q, alpha);
      
      // Compute angular velocity using numerical differentiation of quaternion
      double alpha_plus = std::min(1.0, alpha + eps / (p1.t - p0.t));
      double alpha_minus = std::max(0.0, alpha - eps / (p1.t - p0.t));
      Eigen::Quaterniond q_plus = slerp(p0.q, p1.q, alpha_plus);
      Eigen::Quaterniond q_minus = slerp(p0.q, p1.q, alpha_minus);
      
      double dt_diff = (alpha_plus - alpha_minus) * (p1.t - p0.t);
      Eigen::Quaterniond dq_dt;
      dq_dt.w() = (q_plus.w() - q_minus.w()) / dt_diff;
      dq_dt.x() = (q_plus.x() - q_minus.x()) / dt_diff;
      dq_dt.y() = (q_plus.y() - q_minus.y()) / dt_diff;
      dq_dt.z() = (q_plus.z() - q_minus.z()) / dt_diff;
      
      Eigen::Vector3d gyro_body = compute_angular_velocity(q, dq_dt);
      
      // Specific force in body frame: f_b = R^T * (a_w - g)
      Sophus::SO3d R(q);
      Eigen::Vector3d accel_body = R.inverse() * (acc_world - g);

      // Add noise
      if (enable_noise) {
        for (int i = 0; i < 3; ++i) {
          gyro_body[i]  += gyro_noise_dist(gen);
          accel_body[i] += accel_noise_dist(gen);
        }
        gyro_body  += gyro_bias;
        accel_body += accel_bias;

        const double dt_sqrt = std::sqrt(dt_imu);
        for (int i = 0; i < 3; ++i) {
          gyro_bias[i]  += gyro_bias_dist(gen) * dt_sqrt;
          accel_bias[i] += accel_bias_dist(gen) * dt_sqrt;
        }
      }

      if (sample_count < 5) {
        std::cout << "t=" << std::fixed << std::setprecision(6) << t
                  << " gyro=" << std::setprecision(4) << gyro_body.transpose()
                  << " accel=" << accel_body.transpose()
                  << " |accel|=" << std::setprecision(2) << accel_body.norm() << "\n";
      }

      if (output_to_bag) {
        sensor_msgs::Imu imu_msg;
        imu_msg.header.stamp = ros::Time(t);
        imu_msg.header.frame_id = "/imu";

        imu_msg.orientation.x = q.x();
        imu_msg.orientation.y = q.y();
        imu_msg.orientation.z = q.z();
        imu_msg.orientation.w = q.w();

        imu_msg.angular_velocity.x = gyro_body.x();
        imu_msg.angular_velocity.y = gyro_body.y();
        imu_msg.angular_velocity.z = gyro_body.z();

        imu_msg.linear_acceleration.x = accel_body.x();
        imu_msg.linear_acceleration.y = accel_body.y();
        imu_msg.linear_acceleration.z = accel_body.z();

        for (int i = 0; i < 9; ++i) {
          imu_msg.angular_velocity_covariance[i] = 0.0;
          imu_msg.linear_acceleration_covariance[i] = 0.0;
          imu_msg.orientation_covariance[i] = 0.0;
        }
        imu_msg.angular_velocity_covariance[0] = gyro_noise_std * gyro_noise_std;
        imu_msg.angular_velocity_covariance[4] = gyro_noise_std * gyro_noise_std;
        imu_msg.angular_velocity_covariance[8] = gyro_noise_std * gyro_noise_std;

        imu_msg.linear_acceleration_covariance[0] = accel_noise_std * accel_noise_std;
        imu_msg.linear_acceleration_covariance[4] = accel_noise_std * accel_noise_std;
        imu_msg.linear_acceleration_covariance[8] = accel_noise_std * accel_noise_std;

        bag->write("/simu/imu", imu_msg.header.stamp, imu_msg);
      } else {
        os << std::fixed << std::setprecision(9)
           << t << " "
           << gyro_body.x() << " " << gyro_body.y() << " " << gyro_body.z() << " "
           << accel_body.x() << " " << accel_body.y() << " " << accel_body.z() << " "
           << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
      }

      ++sample_count;
    }

    if (output_to_bag) {
      bag->close();
    } else {
      os.close();
    }

    std::cout << "\n=== Generation Summary ===\n";
    std::cout << "Total samples: " << sample_count << "\n";
    std::cout << "Noise enabled: " << (enable_noise ? "YES" : "NO") << "\n";
    std::cout << "Wrote: " << output_path << "\n";
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
