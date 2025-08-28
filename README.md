# ROS2 Trackers

Main repository for ROS2 trackers.
- Tracker base on opencv nano tracker
- Tracker base on opencv optical flow tracker
- Tracker base on opencv nvidia optical flow tracker

```bash title="build bridge with custom opencv"
colcon build --packages-up-to cv_bridge \
--cmake-args \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  -DCMAKE_INCLUDE_PATH=/usr/include/opencv4/
```

```
ros2 topic pub --once /track_request vision_msgs/msg/Detection2D "{
  header: {
    stamp: {
      sec: 0,
      nanosec: 0
    },
    frame_id: 'camera'
  },
  bbox: {
    center: {
      position: {
        x: 320.0,
        y: 240.0
      },
      theta: 0.0
    },
    size_x: 100.0,
    size_y: 100.0
  },
  id: '',
  results: []
}"
```

---

```bash
docker run -it --rm \
--runtime=nvidia \
nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04 \
/bin/bash
```

```bash
docker run -it --rm \
--runtime=nvidia \
-v /home/user/Download/opencv:opencv \
nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04 \
/bin/bash
```