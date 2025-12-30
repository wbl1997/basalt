/**
 * TUM Pose to IMU Data Generator (Basalt Se3Spline)
 *
 * 解决“周期性不对齐/相位错位”的关键点：
 *  1) 四元数符号连续化（q 与 -q）
 *  2) 5阶样条前后 padding（重复首末位姿各 4 次）
 *  3) 使用 spline.minTimeNs() 做时间对齐：t_spline = t_traj + spline.minTimeNs()
 *
 * Usage:
 *   ./tum_pose_to_imu <input_tum_file> <output_imu_file> [freq] [knot_time_s]
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
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <sophus/se3.hpp>
#include <basalt/spline/se3_spline.h>

#include <rosbag/bag.h>
#include <sensor_msgs/Imu.h>

// Basalt default gravity convention
static const Eigen::Vector3d g(0, 0, -9.81);

// IMU noise params (EuRoC-like)
static const double GYRO_NOISE_DENSITY     = 1.6968e-04; // rad/s/sqrt(Hz)
static const double ACCEL_NOISE_DENSITY    = 2.0000e-3;  // m/s^2/sqrt(Hz)
static const double GYRO_BIAS_RANDOM_WALK  = 1.9393e-05; // rad/s^2/sqrt(Hz)
static const double ACCEL_BIAS_RANDOM_WALK = 3.0000e-03; // m/s^3/sqrt(Hz)

struct TumPose {
  int64_t t_ns{0};
  Sophus::SE3d T_wb; // assume world<-body (T_wb), consistent with your previous usage
};

static void load_tum_trajectory(const std::string& path,
                                std::vector<int64_t>& times_ns,
                                Eigen::aligned_vector<Sophus::SE3d>& poses_wb) {
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

    // Enforce quaternion sign continuity (q and -q represent same rotation)
    if (has_prev && q_prev.dot(q) < 0.0) {
      q.coeffs() *= -1.0;
    }
    q_prev = q;
    has_prev = true;

    const int64_t t_ns = static_cast<int64_t>(t_s * 1e9);
    times_ns.push_back(t_ns);
    poses_wb.emplace_back(Sophus::SE3d(Sophus::SO3d(q), Eigen::Vector3d(tx, ty, tz)));
  }

  if (times_ns.size() < 2) {
    throw std::runtime_error("Trajectory has < 2 poses.");
  }

  // Ensure monotonic increasing times (sort if necessary)
  // If your file is already sorted, this is a no-op.
  std::vector<size_t> idx(times_ns.size());
  for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
  std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return times_ns[a] < times_ns[b]; });

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

  // Remove duplicates/non-increasing (keep first occurrence)
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

  std::cout << "Loaded " << times_ns.size() << " poses from " << path << std::endl;
}

static Sophus::SE3d interpolate_pose(const std::vector<int64_t>& times_ns,
                                    const Eigen::aligned_vector<Sophus::SE3d>& poses,
                                    int64_t query_time_ns) {
  // clamp
  if (query_time_ns <= times_ns.front()) return poses.front();
  if (query_time_ns >= times_ns.back()) return poses.back();

  auto it = std::lower_bound(times_ns.begin(), times_ns.end(), query_time_ns);
  if (it == times_ns.end()) return poses.back();
  if (it == times_ns.begin()) return poses.front();

  const size_t idx2 = static_cast<size_t>(std::distance(times_ns.begin(), it));
  const size_t idx1 = idx2 - 1;

  const double t1 = static_cast<double>(times_ns[idx1]);
  const double t2 = static_cast<double>(times_ns[idx2]);
  const double t  = static_cast<double>(query_time_ns);

  const double alpha = (t - t1) / (t2 - t1);

  // SE3 interpolation via exp(alpha * log(T1^-1 T2))
  const Sophus::SE3d delta = poses[idx1].inverse() * poses[idx2];
  return poses[idx1] * Sophus::SE3d::exp(alpha * delta.log());
}

static bool ends_with(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

int main(int argc, char** argv) {
  try {
    if (argc < 3) {
      std::cerr << "Usage: " << argv[0]
                << " <input_tum_file> <output_imu_file> [freq] [knot_time_s]\n"
                << "  input_tum_file  : TUM (timestamp tx ty tz qx qy qz qw)\n"
                << "  output_imu_file : .txt or .bag\n"
                << "  freq            : IMU frequency Hz (default 200)\n"
                << "  knot_time_s     : spline knot interval seconds (default auto)\n";
      return 1;
    }

    const std::string input_path  = argv[1];
    const std::string output_path = argv[2];
    const int imu_freq            = (argc > 3) ? std::stoi(argv[3]) : 200;
    const double knot_time_s_arg  = (argc > 4) ? std::stod(argv[4]) : 0.0;

    // 1) load
    std::vector<int64_t> times_ns_raw;
    Eigen::aligned_vector<Sophus::SE3d> poses_wb_raw;
    load_tum_trajectory(input_path, times_ns_raw, poses_wb_raw);

    // 2) normalize time to avoid huge nanoseconds
    const int64_t time_offset_raw = times_ns_raw.front();
    std::vector<int64_t> times_ns(times_ns_raw.size());
    for (size_t i = 0; i < times_ns_raw.size(); ++i) times_ns[i] = times_ns_raw[i] - time_offset_raw;
    const int64_t total_duration_ns = times_ns.back() - times_ns.front();

    // 3) knot interval
    // 相位差的主要来源：knot间隔过大会导致样条平滑效应，使得加速度/角速度相位滞后
    // 建议：knot间隔应接近或小于输入位姿的平均间隔
    int64_t dt_knot_ns = 0;
    if (knot_time_s_arg > 0.0) {
      dt_knot_ns = static_cast<int64_t>(knot_time_s_arg * 1e9);
    } else {
      const int64_t avg_interval_ns = total_duration_ns / static_cast<int64_t>(times_ns.size() - 1);
      // 使用更小的knot间隔来减少相位差
      // 设为平均间隔的一半，但限制在 [0.01, 0.2] 秒范围内
      dt_knot_ns = std::max<int64_t>(static_cast<int64_t>(0.01e9),
                                     std::min<int64_t>(static_cast<int64_t>(0.2e9), avg_interval_ns / 2));
    }

    std::cout << "IMU freq: " << imu_freq << " Hz\n";
    std::cout << "Spline knot interval: " << (dt_knot_ns * 1e-9) << " s\n";
    std::cout << "Input pose avg interval: " << (total_duration_ns / (times_ns.size() - 1) * 1e-9) << " s\n";

    // 4) build Se3Spline with padding + time alignment
    // 5阶样条支持到3阶导数，相位误差稍大但更平滑
    constexpr int ORDER = 5;
    constexpr int PAD = ORDER - 1; // 3

    basalt::Se3Spline<ORDER> spline(dt_knot_ns);

    const Sophus::SE3d pose0 = interpolate_pose(times_ns, poses_wb_raw, 0);
    const Sophus::SE3d poseN = interpolate_pose(times_ns, poses_wb_raw, times_ns.back());

    // front padding
    for (int i = 0; i < PAD; ++i) spline.knotsPushBack(pose0);

    // core knots cover [0, total_duration]
    const int num_core_knots = static_cast<int>(total_duration_ns / dt_knot_ns) + 1;
    for (int i = 0; i < num_core_knots; ++i) {
      int64_t knot_t = static_cast<int64_t>(i) * dt_knot_ns;
      if (knot_t > times_ns.back()) knot_t = times_ns.back();
      Sophus::SE3d pose = interpolate_pose(times_ns, poses_wb_raw, knot_t);
      spline.knotsPushBack(pose);
    }

    // back padding
    for (int i = 0; i < PAD; ++i) spline.knotsPushBack(poseN);

    // Time alignment:
    // We will evaluate the spline at t_spline = t_traj + spline.minTimeNs()
    const int64_t spline_t0 = spline.minTimeNs();
    const int64_t spline_t1 = spline.maxTimeNs();

    std::cout << "Spline internal time range: [" << (spline_t0 * 1e-9) << ", " << (spline_t1 * 1e-9) << "] s\n";
    std::cout << "Trajectory time range:      [0, " << (times_ns.back() * 1e-9) << "] s\n";

    // 5) fitting quality check at a few indices
    auto rad2deg = [](double r) { return r * 180.0 / M_PI; };
    std::cout << "\n=== Spline Fitting Quality Check (aligned) ===\n";
    std::vector<size_t> check_idx = {0, times_ns.size() / 3, times_ns.size() * 2 / 3, times_ns.size() - 1};
    for (size_t k : check_idx) {
      int64_t t_traj = times_ns[k];
      int64_t t_spline = t_traj + spline_t0; // align

      if (t_spline < spline.minTimeNs() || t_spline >= spline.maxTimeNs()) {
        std::cout << "k=" << k << " t_traj=" << (t_traj * 1e-9) << " OUTSIDE spline domain\n";
        continue;
      }

      Sophus::SE3d T_gt = poses_wb_raw[k];
      Sophus::SE3d T_sp = spline.pose(t_spline);

      Eigen::Vector3d dp = T_sp.translation() - T_gt.translation();
      Sophus::SO3d dR = T_sp.so3() * T_gt.so3().inverse();
      double ang = dR.log().norm();

      std::cout << "k=" << k
                << " t=" << std::fixed << std::setprecision(6) << ((t_traj + time_offset_raw) * 1e-9)
                << " pos_err=" << std::setprecision(6) << dp.norm() << " m"
                << " rot_err=" << std::setprecision(3) << rad2deg(ang) << " deg\n";
    }
    std::cout << "=============================================\n\n";

    // 6) output setup
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

    // 7) noise model (same scaling you used)
    const double dt_s = 1.0 / static_cast<double>(imu_freq);
    const int64_t dt_ns = static_cast<int64_t>(1e9 / static_cast<double>(imu_freq));
    // 移除 offset 偏移，确保时间戳与采样时刻完全对齐，避免相位差
    // const int64_t offset = dt_ns / 2; // sample at mid-interval like vio_sim

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

    // 8) sampling over trajectory time, aligned to spline time
    const int64_t traj_start = 0;
    const int64_t traj_end   = times_ns.back();

    std::cout << "Sampling IMU from t=" << ((traj_start + time_offset_raw) * 1e-9)
              << " to t=" << ((traj_end + time_offset_raw) * 1e-9)
              << " at " << imu_freq << " Hz\n";

    std::cout << "\n=== First 5 IMU samples ===\n";
    int sample_count = 0;

    // 从 traj_start 开始采样，不加 offset，确保加速度和角速度时间戳完全一致
    for (int64_t t_traj = traj_start; t_traj < traj_end; t_traj += dt_ns) {
      const int64_t t_spline = t_traj + spline_t0;

      // Safety: ensure in spline domain
      if (t_spline < spline.minTimeNs() || t_spline >= spline.maxTimeNs()) continue;

      const Sophus::SE3d T_wb = spline.pose(t_spline);
      const Eigen::Quaterniond q = T_wb.so3().unit_quaternion();

      // Gyro: body-frame angular velocity
      Eigen::Vector3d gyro_body = spline.rotVelBody(t_spline);

      // Accel: specific force in body frame: f_b = R_bw*(a_w - g)
      const Eigen::Vector3d acc_world = spline.transAccelWorld(t_spline);
      Eigen::Vector3d accel_body = T_wb.so3().inverse() * (acc_world - g);

      // Add noise + bias
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

      // restore absolute time
      const double t_abs_s = static_cast<double>(t_traj + time_offset_raw) / 1e9;

      if (sample_count < 5) {
        std::cout << "t=" << std::fixed << std::setprecision(6) << t_abs_s
                  << " gyro=" << std::setprecision(6) << gyro_body.transpose()
                  << " accel=" << accel_body.transpose()
                  << " |accel|=" << std::setprecision(6) << accel_body.norm() << "\n";
      }

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

        // simple diagonal covariances
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
           << t_abs_s << " "
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

