ECE5242 Project 3 - Code Running Instructions

Files required:
- run_slam.py
- load_data.py
- ECE5242Proj3-train/MapUtilsCython/  (for MapUtils_fclad)
- A test data folder containing:
  EncodersXX.mat
  HokuyoXX.mat
  imuXX.mat

Put the test data in:

ECE5242Proj3-test/data/ EncodersXX.mat + HokuyoXX.mat + imuXX.mat

Open a terminal in the project root folder and run:

python run_slam.py --data_id 20

or specify data folder:

python run_slam.py --data_id 20 --data_dir "C:\path\to\ECE5242Proj3-test\data"

# '20' can be replace to other number, like '21' or '23' or actual data number

Output:
The script will generate and display:
1. Occupancy grid map + odometry trajectory
2. Particle filter SLAM map
3. Odometry vs Particle SLAM comparison