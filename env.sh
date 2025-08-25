# sudo sysctl -w net.ipv4.ipfrag_time=3
# sudo sysctl -w net.ipv4.ipfrag_high_thresh=134217728
# sudo sysctl -w net.core.rmem_max=2147483647

# export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# export CYCLONEDDS_URI=file://$PWD/cyclonedds.xml
source /opt/ros/humble/setup.bash
source install/setup.bash
export PYTHONPATH=$PYTHONPATH:/workspace/submodules/of_vio
source aliases.sh