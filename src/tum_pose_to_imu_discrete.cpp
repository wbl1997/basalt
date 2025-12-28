#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <sstream>
#include <random>
#include <cmath>

#include <sophus/se3.hpp>
#include <basalt/utils/sophus_utils.hpp>

#include <rosbag/bag.h>
#include <sensor_msgs/Imu.h>

// 定义重力向量，Basalt 默认为 [0, 0, -9.81]
const Eigen::Vector3d g(0, 0, -9.81);

// // IMU 噪声参数 (参考 Euroc 数据集)
// const double GYRO_NOISE_DENSITY = 1.6968e-04; // rad / s / sqrt(Hz)
// const double ACCEL_NOISE_DENSITY = 2.0000e-3; // m / s^2 / sqrt(Hz)
// const double GYRO_BIAS_RANDOM_WALK = 1.9393e-05; // rad / s^2 / sqrt(Hz)
// const double ACCEL_BIAS_RANDOM_WALK = 3.0000e-03; // m / s^3 / sqrt(Hz)

const double GYRO_NOISE_DENSITY = 1.6968e-09; // rad / s / sqrt(Hz)
const double ACCEL_NOISE_DENSITY = 2.0000e-9; // m / s^2 / sqrt(Hz)
const double GYRO_BIAS_RANDOM_WALK = 1.9393e-09; // rad / s^2 / sqrt(Hz)
const double ACCEL_BIAS_RANDOM_WALK = 3.0000e-09; // m / s^3 / sqrt(Hz)

/**
 * 轨迹加载函数
 * 格式: timestamp(s) tx ty tz qx qy qz qw
 */
void load_tum_trajectory(const std::string& path,
                         std::vector<double>& times,
                         Eigen::aligned_vector<Sophus::SE3d>& poses) {
    std::ifstream is(path);
    std::string line;

    Eigen::Quaterniond q_prev;
    bool has_prev = false;

    while (std::getline(is, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        double t, tx, ty, tz, qx, qy, qz, qw;
        ss >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

        Eigen::Quaterniond q(qw, qx, qy, qz);
        q.normalize();

        // enforce sign continuity: 确保四元数符号连续
        if (has_prev && q_prev.dot(q) < 0) {
            q.coeffs() *= -1.0;
        }
        q_prev = q;
        has_prev = true;

        times.push_back(t);
        poses.emplace_back(Sophus::SE3d(Sophus::SO3d(q), Eigen::Vector3d(tx, ty, tz)));
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./tum_pose_to_imu_discrete <input_tum_file> <output_imu_file> [freq]" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    int imu_freq = (argc > 3) ? std::stoi(argv[3]) : 200; // 默认 200Hz

    // 1. 加载 TUM 数据
    std::vector<double> gt_times;
    Eigen::aligned_vector<Sophus::SE3d> gt_poses;
    load_tum_trajectory(input_path, gt_times, gt_poses);

    if (gt_times.empty()) {
        std::cerr << "Empty trajectory file!" << std::endl;
        return -1;
    }

    // 2. 准备输出和噪声生成器
    bool output_to_bag = (output_path.size() >= 4 && output_path.substr(output_path.size() - 4) == ".bag");
    
    std::ofstream os;
    std::unique_ptr<rosbag::Bag> bag;
    
    if (output_to_bag) {
        bag.reset(new rosbag::Bag);
        bag->open(output_path, rosbag::bagmode::Write);
        std::cout << "Outputting to ROS bag: " << output_path << std::endl;
    } else {
        os.open(output_path);
        os << std::fixed << std::setprecision(9);
        os << "# timestamp(s) wx wy wz ax ay az qx qy qz qw" << std::endl;
        std::cout << "Outputting to text file: " << output_path << std::endl;
    }

    double dt_s = 1.0 / imu_freq;
    
    // 随机数生成器
    std::mt19937 gen(12345);
    
    // 离散时间噪声标准差
    double gyro_noise_std = GYRO_NOISE_DENSITY / std::sqrt(dt_s);
    double accel_noise_std = ACCEL_NOISE_DENSITY / std::sqrt(dt_s);

    std::normal_distribution<> gyro_noise_dist(0, gyro_noise_std);
    std::normal_distribution<> accel_noise_dist(0, accel_noise_std);
    
    // 偏置随机游走分布
    std::normal_distribution<> gyro_bias_dist(0, GYRO_BIAS_RANDOM_WALK);
    std::normal_distribution<> accel_bias_dist(0, ACCEL_BIAS_RANDOM_WALK);

    // 初始偏置
    Eigen::Vector3d current_gyro_bias = Eigen::Vector3d::Random() / 100.0;
    Eigen::Vector3d current_accel_bias = Eigen::Vector3d::Random() / 10.0;

    // 3. 离散差分生成 IMU 数据
    
    size_t current_idx = 0;
    double current_time = gt_times.front();
    double end_time = gt_times.back();

    while (current_time < end_time) {
        // 找到当前时间所在的轨迹段 [idx, idx+1]
        while (current_idx + 1 < gt_times.size() && gt_times[current_idx + 1] < current_time) {
            current_idx++;
        }
        
        if (current_idx + 1 >= gt_times.size()) break;

        double t0 = gt_times[current_idx];
        double t1 = gt_times[current_idx + 1];
        double dt_segment = t1 - t0;
        
        if (dt_segment <= 1e-6) {
            current_time += dt_s;
            continue;
        }

        // 线性插值系数 alpha
        double alpha = (current_time - t0) / dt_segment;

        // 1. 插值位姿 (SE3 插值)
        // T_interp = T0 * exp(log(T0^-1 * T1) * alpha)
        Sophus::SE3d T0 = gt_poses[current_idx];
        Sophus::SE3d T1 = gt_poses[current_idx + 1];
        Sophus::SE3d T_interp = T0 * Sophus::SE3d::exp(alpha * (T0.inverse() * T1).log());
        Eigen::Quaterniond q = T_interp.so3().unit_quaternion();

        // 2. 计算角速度 (Gyro)
        // R(t) = R0 * exp(w_body * t)  =>  w_body = log(R0^T * R1) / dt
        // 这里假设段内角速度恒定
        Eigen::Vector3d w_body_avg = (T0.so3().inverse() * T1.so3()).log() / dt_segment;
        
        // 3. 计算加速度 (Accel)
        // 为了更平滑，我们计算当前段的平均线速度
        Eigen::Vector3d v_w_segment = (T1.translation() - T0.translation()) / dt_segment;
        
        // 计算下一段的平均线速度 (如果存在)
        Eigen::Vector3d v_w_next_segment = v_w_segment;
        if (current_idx + 2 < gt_times.size()) {
            double t2 = gt_times[current_idx + 2];
            v_w_next_segment = (gt_poses[current_idx + 2].translation() - T1.translation()) / (t2 - t1);
        }

        // 计算段间的加速度 (在 t1 时刻)
        double dt_avg = (dt_segment + (current_idx + 2 < gt_times.size() ? (gt_times[current_idx+2] - t1) : dt_segment)) / 2.0;
        Eigen::Vector3d a_w_avg = (v_w_next_segment - v_w_segment) / dt_avg;
        
        Eigen::Vector3d a_w = a_w_avg; 

        // 转换到机体系并减去重力
        // a_imu = R^T * (a_w - g)
        Eigen::Vector3d a_body = T_interp.so3().inverse() * (a_w - g);
        Eigen::Vector3d w_body = w_body_avg;

        // 添加噪声
        for (int i = 0; i < 3; ++i) {
            w_body[i] += gyro_noise_dist(gen);
            a_body[i] += accel_noise_dist(gen);
        }

        // 添加偏置
        w_body += current_gyro_bias;
        a_body += current_accel_bias;

        // 更新偏置 (随机游走)
        double dt_sqrt = std::sqrt(dt_s);
        for (int i = 0; i < 3; ++i) {
            current_gyro_bias[i] += gyro_bias_dist(gen) * dt_sqrt;
            current_accel_bias[i] += accel_bias_dist(gen) * dt_sqrt;
        }

        // 输出
        if (output_to_bag) {
            sensor_msgs::Imu imu_msg;
            imu_msg.header.stamp = ros::Time(current_time);
            imu_msg.header.frame_id = "/imu";
            
            imu_msg.orientation.x = q.x();
            imu_msg.orientation.y = q.y();
            imu_msg.orientation.z = q.z();
            imu_msg.orientation.w = q.w();

            imu_msg.angular_velocity.x = w_body.x();
            imu_msg.angular_velocity.y = w_body.y();
            imu_msg.angular_velocity.z = w_body.z();
            
            imu_msg.linear_acceleration.x = a_body.x();
            imu_msg.linear_acceleration.y = a_body.y();
            imu_msg.linear_acceleration.z = a_body.z();
            
            // 简单的协方差矩阵 (对角线)
            for(int i=0; i<9; i++) {
                imu_msg.angular_velocity_covariance[i] = 0;
                imu_msg.linear_acceleration_covariance[i] = 0;
                imu_msg.orientation_covariance[i] = 0; 
            }
            imu_msg.angular_velocity_covariance[0] = gyro_noise_std * gyro_noise_std;
            imu_msg.angular_velocity_covariance[4] = gyro_noise_std * gyro_noise_std;
            imu_msg.angular_velocity_covariance[8] = gyro_noise_std * gyro_noise_std;
            
            imu_msg.linear_acceleration_covariance[0] = accel_noise_std * accel_noise_std;
            imu_msg.linear_acceleration_covariance[4] = accel_noise_std * accel_noise_std;
            imu_msg.linear_acceleration_covariance[8] = accel_noise_std * accel_noise_std;

            bag->write("/simu/imu", imu_msg.header.stamp, imu_msg);
        } else {
            os << current_time << " "
               << w_body.x() << " " << w_body.y() << " " << w_body.z() << " "
               << a_body.x() << " " << a_body.y() << " " << a_body.z() << " "
               << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
        }

        current_time += dt_s;
    }

    if (output_to_bag) {
        bag->close();
    } else {
        os.close();
    }

    std::cout << "Successfully generated discrete IMU data to: " << output_path << std::endl;
    return 0;
}
