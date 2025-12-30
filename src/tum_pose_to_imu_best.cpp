/**
 * TUM Pose to IMU Data Generator (Final Optimized Version)
 *
 * Features:
 * 1. [Fix Peak Attenuation] Uses Catmull-Rom Cubic Interpolation for translation to restore acceleration peaks.
 * 2. [Fix Phase Lag] Forces 100Hz Knot density regardless of input frequency.
 * 3. [Fix Time Shift] Uses Negative Time Padding to align t=0 perfectly.
 * 4. [Fix Boundary Spikes] Uses Constant Velocity Extrapolation at boundaries.
 *
 * Dependencies: Basalt, Sophus, Eigen3, ROS (roscpp, rosbag, sensor_msgs)
 */

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

// Sophus & Eigen
#include <sophus/se3.hpp>

// Basalt Spline
#include <basalt/spline/se3_spline.h>

// ROS
#include <rosbag/bag.h>
#include <sensor_msgs/Imu.h>
#include <ros/time.h>

// ----------------------------------------------------------------------------
// Configuration
// ----------------------------------------------------------------------------

static const Eigen::Vector3d g_w(0, 0, -9.81); // Gravity in World Frame

// IMU Noise Params (EuRoC)
static const double GYRO_NOISE_DENSITY     = 1.6968e-04; 
static const double ACCEL_NOISE_DENSITY    = 2.0000e-3;  
static const double GYRO_BIAS_RANDOM_WALK  = 1.9393e-05; 
static const double ACCEL_BIAS_RANDOM_WALK = 3.0000e-03; 

// Spline Settings
constexpr int SPLINE_ORDER = 5;          // 5th order for smooth acceleration (C3 continuous)
constexpr int64_t KNOT_DT_NS = 10000000; // 10ms (100Hz) - Crucial for minimizing phase lag

// ----------------------------------------------------------------------------
// Helper Functions
// ----------------------------------------------------------------------------

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

        if (has_prev && q_prev.dot(q) < 0.0) {
            q.coeffs() *= -1.0;
        }
        q_prev = q;
        has_prev = true;

        times_ns.push_back(static_cast<int64_t>(t_s * 1e9));
        poses_wb.emplace_back(Sophus::SE3d(Sophus::SO3d(q), Eigen::Vector3d(tx, ty, tz)));
    }
    
    if (times_ns.size() < 2) throw std::runtime_error("Trajectory too short");
    std::cout << "Loaded " << times_ns.size() << " poses.\n";
}

// Linear Interpolation + Constant Velocity Extrapolation (Fallback)
static Sophus::SE3d interpolate_pose_linear_robust(
    const std::vector<int64_t>& times_ns,
    const Eigen::aligned_vector<Sophus::SE3d>& poses,
    int64_t query_time_ns) {

    if (times_ns.empty()) return Sophus::SE3d();

    // 1. Interpolate
    if (query_time_ns >= times_ns.front() && query_time_ns <= times_ns.back()) {
        auto it = std::lower_bound(times_ns.begin(), times_ns.end(), query_time_ns);
        if (it == times_ns.begin()) return poses.front();

        size_t idx2 = std::distance(times_ns.begin(), it);
        size_t idx1 = idx2 - 1;

        double t1 = (double)times_ns[idx1];
        double t2 = (double)times_ns[idx2];
        double t  = (double)query_time_ns;
        double alpha = (t - t1) / (t2 - t1);
        return poses[idx1] * Sophus::SE3d::exp(alpha * (poses[idx1].inverse() * poses[idx2]).log());
    }

    // 2. Extrapolate Backward
    if (query_time_ns < times_ns.front()) {
        double t0 = (double)times_ns[0];
        double t1 = (double)times_ns[1];
        Sophus::SE3d T_01 = poses[0].inverse() * poses[1];
        Sophus::Vector6d vel = T_01.log() / (t1 - t0); 
        double dt = (double)query_time_ns - t0; 
        return poses[0] * Sophus::SE3d::exp(vel * dt);
    } 

    // 3. Extrapolate Forward
    {
        size_t N = times_ns.size();
        double t_last = (double)times_ns[N-1];
        double t_prev = (double)times_ns[N-2];
        Sophus::SE3d T_prev_last = poses[N-2].inverse() * poses[N-1];
        Sophus::Vector6d vel = T_prev_last.log() / (t_last - t_prev);
        double dt = (double)query_time_ns - t_last;
        return poses[N-1] * Sophus::SE3d::exp(vel * dt);
    }
}

// Cubic Interpolation (Catmull-Rom for Translation, SLERP for Rotation)
static Sophus::SE3d interpolate_pose_cubic(
    const std::vector<int64_t>& times_ns,
    const Eigen::aligned_vector<Sophus::SE3d>& poses,
    int64_t query_time_ns) {

    // Boundary check: use linear fallback if not enough points
    if (times_ns.size() < 4) return interpolate_pose_linear_robust(times_ns, poses, query_time_ns);

    auto it = std::lower_bound(times_ns.begin(), times_ns.end(), query_time_ns);
    
    // If outside or near boundary, use robust linear extrapolation
    if (it == times_ns.begin() || it == times_ns.end()) {
        return interpolate_pose_linear_robust(times_ns, poses, query_time_ns);
    }

    size_t i = std::distance(times_ns.begin(), it) - 1; 
    
    // Catmull-Rom requires P(i-1), P(i), P(i+1), P(i+2)
    if (i == 0 || i >= times_ns.size() - 2) {
        return interpolate_pose_linear_robust(times_ns, poses, query_time_ns);
    }

    // Indices
    int64_t t0 = times_ns[i-1], t1 = times_ns[i], t2 = times_ns[i+1], t3 = times_ns[i+2];
    Eigen::Vector3d p0 = poses[i-1].translation();
    Eigen::Vector3d p1 = poses[i].translation();
    Eigen::Vector3d p2 = poses[i+1].translation();
    Eigen::Vector3d p3 = poses[i+2].translation();

    // Normalized time u in [0, 1] for interval [t1, t2]
    double dt_segment = (t2 - t1) * 1e-9;
    double t_local = (query_time_ns - t1) * 1e-9;
    double u = t_local / dt_segment;

    // Catmull-Rom Tangents
    // m1 = (p2 - p0) / (t2 - t0) * dt_segment
    Eigen::Vector3d m1 = (p2 - p0) / ((t2 - t0) * 1e-9) * dt_segment;
    Eigen::Vector3d m2 = (p3 - p1) / ((t3 - t1) * 1e-9) * dt_segment;

    // Hermite Basis
    double u2 = u * u;
    double u3 = u2 * u;
    double h00 = 2*u3 - 3*u2 + 1;
    double h10 = u3 - 2*u2 + u;
    double h01 = -2*u3 + 3*u2;
    double h11 = u3 - u2;

    // Interpolated Position
    Eigen::Vector3d p_interp = h00*p1 + h10*m1 + h01*p2 + h11*m2;

    // Interpolated Rotation (SLERP is usually sufficient for rotation)
    // We reuse the linear function which does SLERP internally
    Sophus::SE3d T_slerp = interpolate_pose_linear_robust(times_ns, poses, query_time_ns);

    return Sophus::SE3d(T_slerp.so3(), p_interp);
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_tum> <output_path> [imu_freq=200]\n"
                  << "       output_path ending in .bag triggers ROS bag output.\n";
        return 1;
    }

    const std::string input_path = argv[1];
    const std::string output_path = argv[2];
    const int imu_freq = (argc > 3) ? std::stoi(argv[3]) : 200;

    // 1. Load Data
    std::vector<int64_t> times_raw;
    Eigen::aligned_vector<Sophus::SE3d> poses_raw;
    try {
        load_tum_trajectory(input_path, times_raw, poses_raw);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // 2. Normalize Time
    int64_t time_offset = times_raw.front();
    std::vector<int64_t> times_rel;
    times_rel.reserve(times_raw.size());
    for (int64_t t : times_raw) times_rel.push_back(t - time_offset);
    int64_t traj_duration_ns = times_rel.back();

    // 3. Build Spline
    //    Using High Frequency Knots (10ms) + Cubic Interpolation
    std::cout << "Building Spline (Order " << SPLINE_ORDER << ", 10ms Knot, Cubic Interp)...\n";
    basalt::Se3Spline<SPLINE_ORDER> spline(KNOT_DT_NS);

    // Padding Strategy:
    // Spline valid range starts at index (Order-1). 
    // We generate knots starting from negative time so that index (Order-1) is exactly t=0.
    int padding_knots = SPLINE_ORDER - 1;
    int64_t t_start_gen = -(padding_knots * KNOT_DT_NS); 
    int64_t t_end_gen   = traj_duration_ns + (SPLINE_ORDER * KNOT_DT_NS);

    for (int64_t t = t_start_gen; t <= t_end_gen; t += KNOT_DT_NS) {
        // Use CUBIC interpolation here to fix peak attenuation
        Sophus::SE3d pose = interpolate_pose_cubic(times_rel, poses_raw, t);
        spline.knotsPushBack(pose);
    }

    // 4. Output Setup
    bool to_bag = (output_path.size() >= 4 && output_path.substr(output_path.size() - 4) == ".bag");
    std::ofstream ofs;
    std::unique_ptr<rosbag::Bag> bag;

    if (to_bag) {
        bag.reset(new rosbag::Bag);
        bag->open(output_path, rosbag::bagmode::Write);
        std::cout << "Output: ROS Bag -> " << output_path << std::endl;
    } else {
        ofs.open(output_path);
        ofs << std::fixed << std::setprecision(9);
        ofs << "# timestamp(s) wx wy wz ax ay az qx qy qz qw\n";
        std::cout << "Output: Text -> " << output_path << std::endl;
    }

    // 5. Sampling Setup
    std::mt19937 gen(42);
    double dt_s = 1.0 / imu_freq;
    int64_t dt_ns = static_cast<int64_t>(1e9 / imu_freq);

    // Variances
    double var_gyr = pow(GYRO_NOISE_DENSITY / sqrt(dt_s), 2);
    double var_acc = pow(ACCEL_NOISE_DENSITY / sqrt(dt_s), 2);

    std::normal_distribution<double> n_gyr(0.0, sqrt(var_gyr));
    std::normal_distribution<double> n_acc(0.0, sqrt(var_acc));
    std::normal_distribution<double> w_gyr(0.0, GYRO_BIAS_RANDOM_WALK * sqrt(dt_s));
    std::normal_distribution<double> w_acc(0.0, ACCEL_BIAS_RANDOM_WALK * sqrt(dt_s));

    Eigen::Vector3d bg = Eigen::Vector3d::Zero();
    Eigen::Vector3d ba = Eigen::Vector3d::Zero();
    int samples = 0;

    // 6. Sampling Loop
    // Start exactly at 0. Because of our padding strategy, 
    // spline.minTimeNs() should be <= 0, making t=0 valid.
    for (int64_t t_curr = 0; t_curr <= traj_duration_ns; t_curr += dt_ns) {
        
        if (t_curr < spline.minTimeNs() || t_curr >= spline.maxTimeNs()) continue;

        // A. Ground Truth Kinematics from Spline
        Sophus::SE3d T_wb = spline.pose(t_curr);
        Eigen::Vector3d acc_w = spline.transAccelWorld(t_curr);
        Eigen::Vector3d gyr_b = spline.rotVelBody(t_curr);

        // B. Transform to IMU Frame (Body Frame)
        // a_meas = R_bw * (a_world - g_world)
        Eigen::Vector3d acc_b = T_wb.so3().inverse() * (acc_w - g_w);

        // C. Add Noise & Bias
        Eigen::Vector3d gyr_meas = gyr_b + bg + Eigen::Vector3d(n_gyr(gen), n_gyr(gen), n_gyr(gen));
        Eigen::Vector3d acc_meas = acc_b + ba + Eigen::Vector3d(n_acc(gen), n_acc(gen), n_acc(gen));

        // Update Random Walk
        bg += Eigen::Vector3d(w_gyr(gen), w_gyr(gen), w_gyr(gen));
        ba += Eigen::Vector3d(w_acc(gen), w_acc(gen), w_acc(gen));

        double t_abs_s = (time_offset + t_curr) * 1e-9;

        // D. Write Output
        if (to_bag) {
            sensor_msgs::Imu msg;
            msg.header.stamp = ros::Time(t_abs_s);
            msg.header.frame_id = "imu";
            
            Eigen::Quaterniond q = T_wb.unit_quaternion();
            msg.orientation.w = q.w(); msg.orientation.x = q.x(); 
            msg.orientation.y = q.y(); msg.orientation.z = q.z();

            msg.angular_velocity.x = gyr_meas.x();
            msg.angular_velocity.y = gyr_meas.y();
            msg.angular_velocity.z = gyr_meas.z();

            msg.linear_acceleration.x = acc_meas.x();
            msg.linear_acceleration.y = acc_meas.y();
            msg.linear_acceleration.z = acc_meas.z();
            
            // Covariances
            for(int k=0; k<9; ++k) {
                msg.orientation_covariance[k] = 0;
                msg.angular_velocity_covariance[k] = 0;
                msg.linear_acceleration_covariance[k] = 0;
            }
            // Small covariance for orientation (fake GT)
            msg.orientation_covariance[0] = msg.orientation_covariance[4] = msg.orientation_covariance[8] = 1e-6;
            
            // Real noise covariance for accel/gyro
            msg.angular_velocity_covariance[0] = msg.angular_velocity_covariance[4] = msg.angular_velocity_covariance[8] = var_gyr;
            msg.linear_acceleration_covariance[0] = msg.linear_acceleration_covariance[4] = msg.linear_acceleration_covariance[8] = var_acc;

            bag->write("/simu/imu", msg.header.stamp, msg);
        } else {
            Eigen::Quaterniond q = T_wb.unit_quaternion();
            ofs << t_abs_s << " "
                << gyr_meas.x() << " " << gyr_meas.y() << " " << gyr_meas.z() << " "
                << acc_meas.x() << " " << acc_meas.y() << " " << acc_meas.z() << " "
                << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        }
        samples++;
    }

    if (to_bag) bag->close();
    else ofs.close();

    std::cout << "Done. Generated " << samples << " samples.\n";
    return 0;
}