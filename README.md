# AI-Robot-Using-Nvidia-Jetson-Nano
This project is developed on Nvidia Jetson nano Developer Kit 4GB EMMC chip - Eagle 101/J101 Development board

<img width="245" height="490" alt="image" src="https://github.com/user-attachments/assets/fe8dd2c2-b185-4bf8-a441-8183d21ade2a" />

![WhatsApp Image 2025-09-23 at 23 24 52_86ccf9b3](https://github.com/user-attachments/assets/b2585e12-0bca-4667-a3a1-5c2794f8bf36)

<img width="633" height="193" alt="{81DE8B23-D19E-46D6-8DC1-1054C52B2603}" src="https://github.com/user-attachments/assets/95ebf9e9-2809-4a9d-b0fb-5647b35994d0" />






# HardWare Components:
<img width="241" height="209" alt="image" src="https://github.com/user-attachments/assets/0aa8e4cf-9622-4b42-bd5b-2dc8328b6d4f" />
# 1.Nvidia Jetson nano

<img width="960" height="620" alt="image" src="https://github.com/user-attachments/assets/286fb21d-caff-4589-bb0f-27236968f58d" />
# 2.IMX219-83 stereo camera 

<img width="240" height="200" alt="image" src="https://github.com/user-attachments/assets/17298b1d-af17-4b63-8678-c79cf2a68580" />
# 3.Servo Motors




# Step 1:
Boot the Board into SD card 
Check this page if you are using an Eagle 101/J101 Tanna tech balze board to boot it from SD card
"https://medium.com/@anil.sarode_42646/getting-started-with-nvidia-jetson-nano-eagle-101-carrier-board-958bdd5b496f"

# Step 2:
Install all the SDK components with Deepstream SDK using the Nvidia SDK manager (Runs only on Ubuntu Linux Machines)
SDK Manager: "https://developer.nvidia.com/sdk-manager"

# Step 3:
Install ROS Melodic and Build the workspace in an Virtual environment running python 3.8.0
https://wiki.ros.org/melodic/Installation/Ubuntu

# step 4:
Depth perseption using ROS
https://github.com/asujaykk/Stereo-Camera-Depth-Estimation-And-3D-visulaization-on-jetson-nano

# Calibrate the Stereo Cameras
https://github.com/TemugeB/python_stereo_camera_calibrate

<img width="373" height="231" alt="image" src="https://github.com/user-attachments/assets/4ff8eef4-101e-4803-bfad-04051043bfd5" />


# Step 5:
Train the model using the Depth vision with Isaac SIM or pre trained depth SLAM Models

# ISAAC SIM
![WhatsApp Image 2025-10-03 at 16 26 24_88e81c37](https://github.com/user-attachments/assets/be835063-62a6-4f4f-ab45-0e510d3dd255)
![WhatsApp Image 2025-10-03 at 16 26 25_d2c8fa1f](https://github.com/user-attachments/assets/eb6c512a-b72a-4645-a873-5432e9a4803f)
![WhatsApp Image 2025-10-03 at 16 26 27_5edd77a1](https://github.com/user-attachments/assets/a6ce1c98-971f-43e2-bccf-c37bb5bb5fad)

# Action Graph's:
![WhatsApp Image 2025-10-03 at 16 26 28_784a8b43](https://github.com/user-attachments/assets/18f647c9-147e-49c1-91f5-afe3b108e9fb)
![WhatsApp Image 2025-10-03 at 16 26 29_9891bf53](https://github.com/user-attachments/assets/df25145c-4cb2-44d6-9156-a8af2b27b916)

![WhatsApp Image 2025-09-23 at 23 24 48_1dd06ee9](https://github.com/user-attachments/assets/404f2a30-e300-46ee-8904-8278477324e8)


# PRE Trained Models
VINS FUSION : https://github.com/HKUST-Aerial-Robotics/VINS-Fusion
<img width="512" height="248" alt="{C70BFE22-271C-4034-A941-410AA49470CC}" src="https://github.com/user-attachments/assets/7244fb47-091e-4f96-ac56-fc78f3341031" />


https://github.com/user-attachments/assets/e7696048-99b9-49be-877a-20ec72dbfe9c

# Step 6:

Upload the main code






