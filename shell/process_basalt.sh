# /home/wbl/code/rot_liom_new/compare_method/basalt/build/basalt_tum_pose_to_imu <input_tum_file> <output_imu_file> [freq]

# /home/wbl/code/rot_liom_new/compare_method/basalt/build/basalt_tum_pose_to_imu \
#     /home/wbl/code/rot_liom_new/compare_method/basalt/data/tum2imu/fastlio2.txt \
#     /home/wbl/code/rot_liom_new/compare_method/basalt/data/tum2imu/fastlio2_imu_100hz.txt 100

rm -rf /media/wbl/KESU/data/kuangye/shangjia/simu/yiliangloop_imu.bag
/home/wbl/code/rot_liom_new/compare_method/basalt/build/basalt_tum_pose_to_imu_discrete \
    /home/wbl/code/rot_liom_new/experiments/result/ky_output/yiliangloop/fastlio2.txt \
    /media/wbl/KESU/data/kuangye/shangjia/simu/yiliangloop_imu.bag 100
