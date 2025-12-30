/**
 * TUM Pose to IMU Data Generator (Basalt Se3Spline) - Route A + Derivative Switch
 *
 * Route A (smooth knot generation, no global optimization):
 *  - Translation: Catmull-Rom / Hermite cubic (C1)
 *  - Rotation: SQUAD (C1)
 *  - Boundary: constant twist extrapolation
 *
 * Derivative switch:
 *  - analytic: Basalt spline analytic derivatives
 *      gyro_body = spline.rotVelBody(t)
 *      acc_world = spline.transAccelWorld(t)
 *  - numeric: sample pose at imu freq and do finite differences
 *      gyro_body ~ log(R_k^T R_{k+1}) / dt  (last sample uses backward)
 *      acc_world ~ (p_{k+1} - 2 p_k + p_{k-1}) / dt^2 (endpoints one-sided)
 *
 * Usage:
 *   ./tum_pose_to_imu <input_tum_file> <output_imu_file> [freq] [knot_time_s] [derivative_mode]
 *   derivative_mode: analytic | numeric (default: analytic)
 *
 * Input (TUM):
 *   timestamp(s) tx ty tz qx qy qz qw
 *
 * Output:
 *   - .txt: timestamp wx wy wz ax ay az qx qy qz qw
 *   - .bag: sensor_msgs/Imu on topic "/simu/imu"
 */

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <basalt/spline/se3_spline.h>

#include <rosbag/bag.h>
#include <sensor_msgs/Imu.h>
#include <ros/time.h>

// Basalt default gravity convention
static const Eigen::Vector3d g(0, 0, -9.81);

// IMU noise params (EuRoC-like)
static const double GYRO_NOISE_DENSITY     = 1.6968e-04; // rad/s/sqrt(Hz)
static const double ACCEL_NOISE_DENSITY    = 2.0000e-3;  // m/s^2/sqrt(Hz)
static const double GYRO_BIAS_RANDOM_WALK  = 1.9393e-05; // rad/s^2/sqrt(Hz)
static const double ACCEL_BIAS_RANDOM_WALK = 3.0000e-03; // m/s^3/sqrt(Hz)

// =============================
// Helpers: SO(3) log/exp (quat)
// =============================
static inline Eigen::Vector3d so3Log(const Eigen::Quaterniond& q_in) {
  Eigen::Quaterniond q = q_in.normalized();
  if (q.w() < 0.0) q.coeffs() *= -1.0;  // shortest path

  Eigen::Vector3d v(q.x(), q.y(), q.z());
  const double w = q.w();
  const double v_norm = v.norm();

  if (v_norm < 1e-12) {
    return 2.0 * v; // small angle approximation
  }

  const double theta = 2.0 * std::atan2(v_norm, w); // [0, pi]
  const Eigen::Vector3d axis = v / v_norm;
  return theta * axis;
}

static inline Eigen::Quaterniond so3Exp(const Eigen::Vector3d& phi) {
  const double theta = phi.norm();
  Eigen::Quaterniond q;
  if (theta < 1e-12) {
    q.w() = 1.0;
    q.vec() = 0.5 * phi;
    return q.normalized();
  }
  const Eigen::Vector3d axis = phi / theta;
  const double half = 0.5 * theta;
  q.w() = std::cos(half);
  q.vec() = axis * std::sin(half);
  return q.normalized();
}

static inline Eigen::Quaterniond quatSlerpShortest(const Eigen::Quaterniond& q1_in,
                                                   const Eigen::Quaterniond& q2_in,
                                                   double u) {
  Eigen::Quaterniond q1 = q1_in.normalized();
  Eigen::Quaterniond q2 = q2_in.normalized();
  if (q1.dot(q2) < 0.0) q2.coeffs() *= -1.0;
  return q1.slerp(u, q2).normalized();
}

// =============================
// SQUAD (C1) rotation spline
// =============================
static inline Eigen::Quaterniond squadTangent(const Eigen::Quaterniond& q_im1,
                                              const Eigen::Quaterniond& q_i,
                                              const Eigen::Quaterniond& q_ip1) {
  Eigen::Quaterniond qm1 = q_im1.normalized();
  Eigen::Quaterniond qi  = q_i.normalized();
  Eigen::Quaterniond qp1 = q_ip1.normalized();
  if (qi.dot(qm1) < 0.0) qm1.coeffs() *= -1.0;
  if (qi.dot(qp1) < 0.0) qp1.coeffs() *= -1.0;

  Eigen::Quaterniond qi_inv = qi.conjugate();
  Eigen::Quaterniond a = qi_inv * qp1;
  Eigen::Quaterniond b = qi_inv * qm1;

  Eigen::Vector3d log_a = so3Log(a);
  Eigen::Vector3d log_b = so3Log(b);

  Eigen::Vector3d phi = -0.25 * (log_a + log_b);
  return (qi * so3Exp(phi)).normalized();
}

static inline Eigen::Quaterniond squad(const Eigen::Quaterniond& q0,
                                       const Eigen::Quaterniond& s0,
                                       const Eigen::Quaterniond& s1,
                                       const Eigen::Quaterniond& q1,
                                       double u) {
  Eigen::Quaterniond qa = quatSlerpShortest(q0, q1, u);
  Eigen::Quaterniond qb = quatSlerpShortest(s0, s1, u);
  const double h = 2.0 * u * (1.0 - u);
  return quatSlerpShortest(qa, qb, h);
}

// =============================
// Catmull-Rom (Hermite) for translation (C1)
// =============================
static inline Eigen::Vector3d catmullRomTranslation(
    int64_t t0, const Eigen::Vector3d& p0,
    int64_t t1, const Eigen::Vector3d& p1,
    int64_t t2, const Eigen::Vector3d& p2,
    int64_t t3, const Eigen::Vector3d& p3,
    int64_t t_query) {

  const double T0 = t0 * 1e-9;
  const double T1 = t1 * 1e-9;
  const double T2 = t2 * 1e-9;
  const double T3 = t3 * 1e-9;
  const double T  = t_query * 1e-9;

  const double dt = (T2 - T1);
  if (dt <= 1e-12) return p1;

  double u = (T - T1) / dt;
  u = std::min(1.0, std::max(0.0, u));

  Eigen::Vector3d m1 = (p2 - p0) / std::max(1e-12, (T2 - T0));
  Eigen::Vector3d m2 = (p3 - p1) / std::max(1e-12, (T3 - T1));

  const double u2 = u*u;
  const double u3 = u2*u;
  const double h00 =  2*u3 - 3*u2 + 1;
  const double h10 =      u3 - 2*u2 + u;
  const double h01 = -2*u3 + 3*u2;
  const double h11 =      u3 -   u2;

  Eigen::Vector3d t1v = m1 * dt;
  Eigen::Vector3d t2v = m2 * dt;

  return h00*p1 + h10*t1v + h01*p2 + h11*t2v;
}

// =============================
// SE(3) boundary extrapolation: constant twist
// =============================
static inline Sophus::SE3d extrapolate_se3_const_twist(
    const Sophus::SE3d& T0, const Sophus::SE3d& T1,
    int64_t t0_ns, int64_t t1_ns, int64_t t_query_ns) {

  const double t0 = t0_ns * 1e-9;
  const double t1 = t1_ns * 1e-9;
  const double tq = t_query_ns * 1e-9;

  const double dt = std::max(1e-12, (t1 - t0));
  Sophus::SE3d dT = T0.inverse() * T1;
  Sophus::Vector6d xi = dT.log() / dt;   // twist per second
  const double dq = (tq - t0);
  return T0 * Sophus::SE3d::exp(xi * dq);
}

// =============================
// Smooth pose interpolation (Route A)
// =============================
static Sophus::SE3d interpolate_pose_smooth(
    const std::vector<int64_t>& times_ns,
    const Eigen::aligned_vector<Sophus::SE3d>& poses,
    int64_t query_time_ns) {

  const size_t N = times_ns.size();
  if (N == 0) return Sophus::SE3d();
  if (N == 1) return poses[0];

  // Boundary extrapolation
  if (query_time_ns <= times_ns.front()) {
    return extrapolate_se3_const_twist(poses[0], poses[1], times_ns[0], times_ns[1], query_time_ns);
  }
  if (query_time_ns >= times_ns.back()) {
    return extrapolate_se3_const_twist(poses[N-2], poses[N-1], times_ns[N-2], times_ns[N-1], query_time_ns);
  }

  // Segment [i0, i1]
  auto it = std::lower_bound(times_ns.begin(), times_ns.end(), query_time_ns);
  size_t i1 = static_cast<size_t>(std::distance(times_ns.begin(), it));
  if (i1 == 0) i1 = 1;
  size_t i0 = i1 - 1;

  // Need 4 points for Catmull-Rom/SQUAD
  if (i0 == 0 || i1 + 1 >= N) {
    // Fallback: local SE3 geodesic interpolation
    const double t0 = times_ns[i0] * 1e-9;
    const double t1 = times_ns[i1] * 1e-9;
    const double tq = query_time_ns * 1e-9;
    const double alpha = (tq - t0) / std::max(1e-12, (t1 - t0));
    Sophus::SE3d delta = poses[i0].inverse() * poses[i1];
    return poses[i0] * Sophus::SE3d::exp(alpha * delta.log());
  }

  const size_t im1 = i0 - 1;
  const size_t ip2 = i1 + 1;

  // Translation
  const int64_t t0 = times_ns[im1], t1 = times_ns[i0], t2 = times_ns[i1], t3 = times_ns[ip2];
  const Eigen::Vector3d p0 = poses[im1].translation();
  const Eigen::Vector3d p1 = poses[i0].translation();
  const Eigen::Vector3d p2 = poses[i1].translation();
  const Eigen::Vector3d p3 = poses[ip2].translation();
  Eigen::Vector3d p = catmullRomTranslation(t0,p0, t1,p1, t2,p2, t3,p3, query_time_ns);

  // Rotation (SQUAD)
  Eigen::Quaterniond q_im1 = poses[im1].so3().unit_quaternion();
  Eigen::Quaterniond q_i   = poses[i0].so3().unit_quaternion();
  Eigen::Quaterniond q_ip1 = poses[i1].so3().unit_quaternion();
  Eigen::Quaterniond q_ip2 = poses[ip2].so3().unit_quaternion();

  if (q_i.dot(q_im1) < 0.0) q_im1.coeffs() *= -1.0;
  if (q_i.dot(q_ip1) < 0.0) q_ip1.coeffs() *= -1.0;
  if (q_ip1.dot(q_ip2) < 0.0) q_ip2.coeffs() *= -1.0;

  Eigen::Quaterniond s_i   = squadTangent(q_im1, q_i, q_ip1);
  Eigen::Quaterniond s_ip1 = squadTangent(q_i, q_ip1, q_ip2);

  const double T1 = times_ns[i0] * 1e-9;
  const double T2 = times_ns[i1] * 1e-9;
  const double Tq = query_time_ns * 1e-9;
  double u = (Tq - T1) / std::max(1e-12, (T2 - T1));
  u = std::min(1.0, std::max(0.0, u));

  Eigen::Quaterniond q = squad(q_i, s_i, s_ip1, q_ip1, u);
  return Sophus::SE3d(Sophus::SO3d(q), p);
}

// =============================
// I/O & utilities
// =============================
static void load_tum_trajectory(const std::string& path,
                                std::vector<int64_t>& times_ns,
                                Eigen::aligned_vector<Sophus::SE3d>& poses_wb) {
  std::ifstream is(path);
  if (!is.is_open()) throw std::runtime_error("Failed to open: " + path);

  std::string line;
  Eigen::Quaterniond q_prev;
  bool has_prev = false;

  times_ns.clear();
  poses_wb.clear();

  while (std::getline(is, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::stringstream ss(line);
    double t_s, tx, ty, tz, qx, qy, qz, qw;
    if (!(ss >> t_s >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) continue;

    Eigen::Quaterniond q(qw, qx, qy, qz);
    q.normalize();

    if (has_prev && q_prev.dot(q) < 0.0) q.coeffs() *= -1.0;
    q_prev = q;
    has_prev = true;

    times_ns.push_back(static_cast<int64_t>(t_s * 1e9));
    poses_wb.emplace_back(Sophus::SE3d(Sophus::SO3d(q), Eigen::Vector3d(tx, ty, tz)));
  }

  if (times_ns.size() < 2) throw std::runtime_error("Trajectory has < 2 poses.");

  // Sort by time
  std::vector<size_t> idx(times_ns.size());
  for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
  std::sort(idx.begin(), idx.end(),
            [&](size_t a, size_t b) { return times_ns[a] < times_ns[b]; });

  std::vector<int64_t> times_sorted;
  times_sorted.reserve(times_ns.size());
  Eigen::aligned_vector<Sophus::SE3d> poses_sorted;
  poses_sorted.reserve(poses_wb.size());

  for (size_t k : idx) {
    times_sorted.push_back(times_ns[k]);
    poses_sorted.push_back(poses_wb[k]);
  }
  times_ns.swap(times_sorted);
  poses_wb.swap(poses_sorted);

  // Remove duplicates/non-increasing
  std::vector<int64_t> times_clean;
  Eigen::aligned_vector<Sophus::SE3d> poses_clean;
  times_clean.reserve(times_ns.size());
  poses_clean.reserve(poses_wb.size());

  int64_t last_t = std::numeric_limits<int64_t>::min();
  for (size_t i = 0; i < times_ns.size(); ++i) {
    if (times_ns[i] <= last_t) continue;
    last_t = times_ns[i];
    times_clean.push_back(times_ns[i]);
    poses_clean.push_back(poses_wb[i]);
  }
  times_ns.swap(times_clean);
  poses_wb.swap(poses_clean);

  std::cout << "Loaded " << times_ns.size() << " poses from " << path << "\n";
}

static bool ends_with(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

enum class DerivativeMode { Analytic, Numeric };

static DerivativeMode parse_derivative_mode(const std::string& s) {
  if (s == "analytic" || s == "ANALYTIC") return DerivativeMode::Analytic;
  if (s == "numeric"  || s == "NUMERIC")  return DerivativeMode::Numeric;
  throw std::runtime_error("Unknown derivative_mode: " + s + " (use analytic|numeric)");
}

// =============================
// Main
// =============================
int main(int argc, char** argv) {
  try {
    if (argc < 3) {
      std::cerr
        << "Usage: " << argv[0]
        << " <input_tum_file> <output_imu_file> [freq] [knot_time_s] [derivative_mode]\n"
        << "  freq            : IMU frequency Hz (default 200)\n"
        << "  knot_time_s     : spline knot interval seconds (default auto)\n"
        << "  derivative_mode : analytic | numeric (default analytic)\n";
      return 1;
    }

    const std::string input_path  = argv[1];
    const std::string output_path = argv[2];
    const int imu_freq            = (argc > 3) ? std::stoi(argv[3]) : 200;
    const double knot_time_s_arg  = (argc > 4) ? std::stod(argv[4]) : 0.0;
    const std::string mode_str    = (argc > 5) ? std::string(argv[5]) : "analytic";
    const DerivativeMode mode     = parse_derivative_mode(mode_str);

    // 1) load
    std::vector<int64_t> times_ns_raw;
    Eigen::aligned_vector<Sophus::SE3d> poses_wb_raw;
    load_tum_trajectory(input_path, times_ns_raw, poses_wb_raw);

    // 2) normalize time
    const int64_t time_offset_raw = times_ns_raw.front();
    std::vector<int64_t> times_ns(times_ns_raw.size());
    for (size_t i = 0; i < times_ns_raw.size(); ++i) times_ns[i] = times_ns_raw[i] - time_offset_raw;
    const int64_t total_duration_ns = times_ns.back() - times_ns.front();

    // 3) knot interval
    int64_t dt_knot_ns = 0;
    if (knot_time_s_arg > 0.0) {
      dt_knot_ns = static_cast<int64_t>(knot_time_s_arg * 1e9);
    } else {
      const int64_t avg_interval_ns = total_duration_ns / static_cast<int64_t>(times_ns.size() - 1);
      dt_knot_ns = std::max<int64_t>(static_cast<int64_t>(0.01e9),
                                     std::min<int64_t>(static_cast<int64_t>(0.2e9), avg_interval_ns / 2));
    }

    std::cout << "IMU freq: " << imu_freq << " Hz\n";
    std::cout << "Spline knot interval: " << (dt_knot_ns * 1e-9) << " s\n";
    std::cout << "Derivative mode: " << mode_str << "\n";

    // 4) build Se3Spline with padding + time alignment
    constexpr int ORDER = 5;
    constexpr int PAD = ORDER - 1; // =4
    basalt::Se3Spline<ORDER> spline(dt_knot_ns);

    const Sophus::SE3d pose0 = interpolate_pose_smooth(times_ns, poses_wb_raw, 0);
    const Sophus::SE3d poseN = interpolate_pose_smooth(times_ns, poses_wb_raw, times_ns.back());

    for (int i = 0; i < PAD; ++i) spline.knotsPushBack(pose0);

    const int num_core_knots = static_cast<int>(total_duration_ns / dt_knot_ns) + 1;
    for (int i = 0; i < num_core_knots; ++i) {
      int64_t knot_t = static_cast<int64_t>(i) * dt_knot_ns;
      if (knot_t > times_ns.back()) knot_t = times_ns.back();
      spline.knotsPushBack(interpolate_pose_smooth(times_ns, poses_wb_raw, knot_t));
    }

    for (int i = 0; i < PAD; ++i) spline.knotsPushBack(poseN);

    const int64_t spline_t0 = spline.minTimeNs();
    const int64_t spline_t1 = spline.maxTimeNs();

    std::cout << "spline_t0: " << spline_t0 << "\n";
    std::cout << "Spline internal time range: ["
              << (spline_t0 * 1e-9) << ", " << (spline_t1 * 1e-9) << "] s\n";
    std::cout << "Trajectory time range:      [0, " << (times_ns.back() * 1e-9) << "] s\n";

    // 5) output setup
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

    // 6) noise model
    const double dt_s = 1.0 / static_cast<double>(imu_freq);
    const int64_t dt_ns = static_cast<int64_t>(1e9 / static_cast<double>(imu_freq));

    std::mt19937 gen{1};

    const double gyro_noise_std  = GYRO_NOISE_DENSITY / std::sqrt(dt_s);
    const double accel_noise_std = ACCEL_NOISE_DENSITY / std::sqrt(dt_s);

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

    // 7) Sampling timeline
    const int64_t traj_start = 0;
    const int64_t traj_end   = times_ns.back();
    const int num_samples = static_cast<int>(traj_end / dt_ns) + 1;

    std::cout << "Sampling IMU from t=" << ((traj_start + time_offset_raw) * 1e-9)
              << " to t=" << ((traj_end + time_offset_raw) * 1e-9)
              << " at " << imu_freq << " Hz\n";

    // =============================
    // Numeric derivative pre-sampling (pose-only)
    // =============================
    std::vector<int64_t> t_spline_vec;
    std::vector<double> t_abs_vec;
    std::vector<Eigen::Vector3d> p_w_vec;
    std::vector<Eigen::Quaterniond> q_wb_vec;

    t_spline_vec.reserve(num_samples);
    t_abs_vec.reserve(num_samples);
    p_w_vec.reserve(num_samples);
    q_wb_vec.reserve(num_samples);

    if (mode == DerivativeMode::Numeric) {
      for (int64_t t_traj = traj_start; t_traj <= traj_end; t_traj += dt_ns) {
        const int64_t t_spline = t_traj + spline_t0;
        if (t_spline < spline.minTimeNs() || t_spline >= spline.maxTimeNs()) continue;

        const Sophus::SE3d T_wb = spline.pose(t_spline);
        t_spline_vec.push_back(t_spline);
        t_abs_vec.push_back(static_cast<double>(t_traj + time_offset_raw) / 1e9);
        p_w_vec.push_back(T_wb.translation());
        q_wb_vec.push_back(T_wb.so3().unit_quaternion());
      }

      if (p_w_vec.size() < 3) {
        throw std::runtime_error("Numeric mode requires at least 3 valid samples in spline domain.");
      }
    }

    std::cout << "\n=== First 5 IMU samples ===\n";
    int sample_count = 0;

    // =============================
    // Main output loop
    // =============================
    auto write_sample = [&](double t_abs_s,
                            const Eigen::Quaterniond& q,
                            const Eigen::Vector3d& gyro_body,
                            const Eigen::Vector3d& accel_body,
                            double var_gyr,
                            double var_acc) {
      if (output_to_bag) {
        sensor_msgs::Imu imu_msg;
        imu_msg.header.stamp = ros::Time(t_abs_s);
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
        imu_msg.angular_velocity_covariance[0] = var_gyr;
        imu_msg.angular_velocity_covariance[4] = var_gyr;
        imu_msg.angular_velocity_covariance[8] = var_gyr;

        imu_msg.linear_acceleration_covariance[0] = var_acc;
        imu_msg.linear_acceleration_covariance[4] = var_acc;
        imu_msg.linear_acceleration_covariance[8] = var_acc;

        bag->write("/simu/imu", imu_msg.header.stamp, imu_msg);
      } else {
        os << std::fixed << std::setprecision(9)
           << t_abs_s << " "
           << gyro_body.x() << " " << gyro_body.y() << " " << gyro_body.z() << " "
           << accel_body.x() << " " << accel_body.y() << " " << accel_body.z() << " "
           << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
      }
    };

    const double var_gyr = gyro_noise_std * gyro_noise_std;
    const double var_acc = accel_noise_std * accel_noise_std;

    if (mode == DerivativeMode::Analytic) {
      // --- Analytic derivatives (original method) ---
      for (int64_t t_traj = traj_start; t_traj < traj_end; t_traj += dt_ns) {
        const int64_t t_spline = t_traj + spline_t0;
        if (t_spline < spline.minTimeNs() || t_spline >= spline.maxTimeNs()) continue;

        const Sophus::SE3d T_wb = spline.pose(t_spline);
        const Eigen::Quaterniond q = T_wb.so3().unit_quaternion();

        Eigen::Vector3d gyro_body = spline.rotVelBody(t_spline);
        const Eigen::Vector3d acc_world = spline.transAccelWorld(t_spline);
        Eigen::Vector3d accel_body = T_wb.so3().inverse() * (acc_world - g);

        // noise + bias
        if (enable_noise) {
          for (int i = 0; i < 3; ++i) {
            gyro_body[i]  += gyro_noise_dist(gen);
            accel_body[i] += accel_noise_dist(gen);
          }
          gyro_body  += gyro_bias;
          accel_body += accel_bias;

          const double dt_sqrt = std::sqrt(dt_s);
          for (int i = 0; i < 3; ++i) {
            gyro_bias[i]  += gyro_bias_dist(gen) * dt_sqrt;
            accel_bias[i] += accel_bias_dist(gen) * dt_sqrt;
          }
        }

        const double t_abs_s = static_cast<double>(t_traj + time_offset_raw) / 1e9;

        if (sample_count < 5) {
          std::cout << "t=" << std::fixed << std::setprecision(6) << t_abs_s
                    << " gyro=" << std::setprecision(6) << gyro_body.transpose()
                    << " accel=" << accel_body.transpose()
                    << " |accel|=" << std::setprecision(6) << accel_body.norm() << "\n";
        }

        write_sample(t_abs_s, q, gyro_body, accel_body, var_gyr, var_acc);
        ++sample_count;
      }
    } else {
      // --- Numeric derivatives (pose-only sampling + differences) ---
      const int M = static_cast<int>(p_w_vec.size());
      for (int k = 0; k < M; ++k) {
        const double t_abs_s = t_abs_vec[k];

        // const Eigen::Vector3d& p_k = p_w_vec[k];
        const Eigen::Quaterniond& q_k = q_wb_vec[k];
        const Eigen::Matrix3d R_k = q_k.toRotationMatrix();

        // gyro_body: use forward difference at k, backward at last
        Eigen::Vector3d gyro_body = Eigen::Vector3d::Zero();
        if (k < M - 1) {
          Eigen::Quaterniond q_next = q_wb_vec[k+1];
          if (q_k.dot(q_next) < 0.0) q_next.coeffs() *= -1.0;
          Eigen::Quaterniond dq = q_k.conjugate() * q_next;  // body-k frame
          gyro_body = so3Log(dq) / dt_s;
        } else {
          Eigen::Quaterniond q_prev = q_wb_vec[k-1];
          if (q_prev.dot(q_k) < 0.0) q_prev.coeffs() *= -1.0;
          Eigen::Quaterniond dq = q_prev.conjugate() * q_k;  // body-(k-1) frame approx
          gyro_body = so3Log(dq) / dt_s;
        }

        // acc_world: second difference (central), endpoints one-sided
        Eigen::Vector3d acc_world = Eigen::Vector3d::Zero();
        if (k >= 1 && k <= M - 2) {
          acc_world = (p_w_vec[k+1] - 2.0*p_w_vec[k] + p_w_vec[k-1]) / (dt_s*dt_s);
        } else if (k == 0) {
          // one-sided: p0,p1,p2
          acc_world = (p_w_vec[2] - 2.0*p_w_vec[1] + p_w_vec[0]) / (dt_s*dt_s);
        } else { // k == M-1
          acc_world = (p_w_vec[M-1] - 2.0*p_w_vec[M-2] + p_w_vec[M-3]) / (dt_s*dt_s);
        }

        // specific force in body frame
        Eigen::Vector3d accel_body = R_k.transpose() * (acc_world - g);

        // noise + bias
        if (enable_noise) {
          for (int i = 0; i < 3; ++i) {
            gyro_body[i]  += gyro_noise_dist(gen);
            accel_body[i] += accel_noise_dist(gen);
          }
          gyro_body  += gyro_bias;
          accel_body += accel_bias;

          const double dt_sqrt = std::sqrt(dt_s);
          for (int i = 0; i < 3; ++i) {
            gyro_bias[i]  += gyro_bias_dist(gen) * dt_sqrt;
            accel_bias[i] += accel_bias_dist(gen) * dt_sqrt;
          }
        }

        if (sample_count < 5) {
          std::cout << "t=" << std::fixed << std::setprecision(6) << t_abs_s
                    << " gyro=" << std::setprecision(6) << gyro_body.transpose()
                    << " accel=" << accel_body.transpose()
                    << " |accel|=" << std::setprecision(6) << accel_body.norm() << "\n";
        }

        write_sample(t_abs_s, q_k, gyro_body, accel_body, var_gyr, var_acc);
        ++sample_count;
      }
    }

    if (output_to_bag) bag->close();
    else os.close();

    std::cout << "\n=== Generation Summary ===\n";
    std::cout << "Total samples: " << sample_count << "\n";
    std::cout << "Noise enabled: " << (enable_noise ? "YES" : "NO") << "\n";
    std::cout << "Wrote: " << output_path << "\n";
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
