# lidar-odometry-slam

A simple 2D SLAM project using wheel encoder odometry, 2D lidar scans, occupancy grid mapping, and a particle filter.


<img src="assets/overview_img.png" width="450" />

<br>

## Repository Structure

```text
├── run_slam.py                  # main script
├── README.md
├── assets/                  # result images for README
├── ECE5242Proj3-train/
│   ├── load_data.py             # provided data loader
│   ├── test_load_data.py        # loader test / lidar visualization
│   ├── docs/                    # provided project documentation
│   │   ├── platform_config.pdf
│   │   ├── README_Encoders.txt
│   │   └── suggestions.txt
│   ├── MapUtils/                # Python map utilities
│   │   ├── MapUtils.py
│   │   └── test_MapUtils.py
│   ├── MapUtilsCython/          # Cython ray-tracing utility
│   │   ├── setup.py
│   │   ├── readme.txt
│   │   ├── MapUtils_fclad.pyx
│   │   └── ...
│   └── data/                    # training data
│       ├── EncodersXX.mat
│       ├── HokuyoXX.mat
│       ├── imuXX.mat
│       └── mapXX.png
└── ECE5242Proj3-test/
    └── data/                    # test data
        ├── EncodersXX.mat
        ├── HokuyoXX.mat
        ├── imuXX.mat
        └── mapXX.png
```


This project estimates a robot trajectory and builds a 2D occupancy grid map from:

- wheel encoder measurements
- 2D lidar scans


The pipeline includes:

1. **Odometry-only trajectory estimation** from wheel encoder counts
2. **Occupancy grid mapping** using lidar scans and odometry poses
3. **Particle-filter SLAM** for pose correction and map refinement
4. Comparison between odometry-only and SLAM trajectories

<br>

## Methods

### 1. Odometry
Wheel encoder counts are converted to wheel travel distance, then integrated with a differential-drive style motion model to estimate robot pose over time.

<img src="assets/platform_config_overview.png" width="500" />

- `counts_per_revolution = 360.0` # count per revolution (circle)
- `wheel_diameter = 0.254` # meters (254mm)
- `track_width = 0.735` # meters (735mm) tunned

<br>

**Robot increment motion**
<div>
<img src="assets/tracking angular movement.png" width="400" /> <img src="assets/tracking translational motion.png" width="400" />
</div>
<br>

```
d_forward = 0.5*(left_side_distance + right_side_distance)
d_theta = (right_side_distance - left_side_distance) / track_width
```

<br>

**Example maps and odometry-only trajectories**
<div>
  <img src="ECE5242Proj3-train/data/map20.png" width="300" />
  <img src="ECE5242Proj3-train/data/map23.png" width="300" />
</div>

<div>
  <img src="assets/results/map_20_odometry_only_trajectory.png" width="290" />
  <img src="assets/results/map_23_odometry_only_trajectory.png" width="310" />
</div>


### 2. Occupancy Grid Mapping
Lidar endpoints are projected into the world frame using the current robot pose.  

<div>
  <img src="assets/Map Registration.png" width="300" />
  <img src="assets/log-odd.png" width="300" />
</div>

An occupancy grid map is updated with:

- positive log-odds for hit cells
- negative log-odds for free cells along each ray

- `hit_log_odds = 0.7` # more like obstacle wall, more black
- `miss_log_odds = 0.2` # more like free space, more white
- `log_odds_cap = 5.0` # keep log-odds finite, not infinite

<br>

### 3. Particle Filter SLAM

<div>
  <img src="assets/particle filter.png" width="550" />
  <img src="assets/cumulative distribution function.png" width="300" />
</div>

A particle filter is used to improve pose estimation:

- predict particles using odometry increments
- update particle weights using lidar-to-map consistency
- select the highest-weight particle as the SLAM pose estimate
- update the occupancy grid map using the best particle pose

SLAM parameters:
- `num_particles = 100`
- `forward_noise_std = 0.008`
- `theta_noise_std = 0.0025`

<br>

### 4. Results
The final output includes:

- **occupancy grid map from odometry**
<div>
  <img src="assets/results/map_20_occupancy_grid_map+odometry_trajectory.png" width="250" />
  <img src="assets/results/map_21_occupancy_grid_map+odometry_trajectory.png" width="250" />
  <img src="assets/results/map_22_occupancy_grid_map+odometry_trajectory.png" width="250" />
  <img src="assets/results/map_23_occupancy_grid_map+odometry_trajectory.png" width="250" />
  <img src="assets/results/map_24_occupancy_grid_map+odometry_trajectory.png" width="250" />
</div>
<br>

- **particle-filter SLAM map**
<div>
  <img src="assets/results/map_20_particle_filter_SLAM_map.png" width="250" />
  <img src="assets/results/map_21_particle_filter_SLAM_map.png" width="250" />
  <img src="assets/results/map_22_particle_filter_SLAM_map.png" width="230" />
  <img src="assets/results/map_23_particle_filter_SLAM_map.png" width="250" />
  <img src="assets/results/map_24_particle_filter_SLAM_map.png" width="250" />
</div>
<br>

- **comparison between odometry-only and SLAM trajectories**
<div>
  <img src="assets/results/map_20_odometry_vs_particle_SLAM.png" width="250" />
  <img src="assets/results/map_21_odometry_vs_particle_SLAM.png" width="250" />
  <img src="assets/results/map_22_odometry_vs_particle_SLAM.png" width="230" />
  <img src="assets/results/map_23_odometry_vs_particle_SLAM.png" width="250" />
  <img src="assets/results/map_24_odometry_vs_particle_SLAM.png" width="250" />
</div>
<br>

### Notes
- The blue SLAM trajectory corresponds to the highest-weight particle over time.
- The code depends on the provided load_data.py and MapUtilsCython files.
- Training data is not intended to be included in the final submission.
- The script is designed to run on a new test dataset by changing the input data folder and dataset ID.

<br>

## How to Run
1. Make sure the required project files are available:
   - `load_data.py`
   - `ECE5242Proj3-train/MapUtilsCython`
2. Put the dataset in the expected data folder `ECE5242Proj3-test/data`.
3. Run the main script with a dataset ID. 

In the 'Lidar-Odometry-SLAM' directory, run
```bash
python run_slam.py --data_id 20
```
or
```bash
python run_slam.py --data_id 20 --data_dir "your path"
```
This can be data_id 22, 24, or your renamed data with data_id

And this will generate: odometry map, particle SLAM map, and odometry vs SLAM comparison
