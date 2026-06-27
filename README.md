# WiFi CSI-Based Indoor Localization and Sensing for Drone Navigation

## Overview

This repository contains my implementation and exploration of **WiFi Channel State Information (CSI)** for indoor sensing, localization, gesture recognition, and room imaging. The project was completed as part of an assignment investigating whether WiFi CSI can provide an alternative sensing modality for autonomous drones operating in GPS-denied environments.

Traditional drone navigation depends heavily on cameras, LiDAR, GPS, and IMUs. While these sensors perform well under normal conditions, they often struggle indoors, in smoke-filled environments, dusty warehouses, or low-light conditions. WiFi CSI provides a different sensing mechanism by exploiting the way radio signals propagate through an environment.

Unlike Received Signal Strength Indicator (RSSI), CSI captures the complex wireless channel response for every OFDM subcarrier, allowing detailed observation of multipath propagation, reflections, attenuation, and Doppler effects. This rich information has enabled recent research in indoor localization, activity recognition, gesture recognition, occupancy sensing, and environmental imaging.

The objective of this project was to:

* Study recent WiFi CSI localization and sensing literature.
* Understand the complete CSI processing pipeline.
* Implement multiple CSI processing techniques inspired by published research.
* Evaluate machine learning and deep learning models using public CSI datasets.
* Investigate how these techniques could be extended toward autonomous drone navigation.

---

# Repository Structure

```text
.
    - code/
        - feature_extraction_gesture.ipynb
        - gesture_cnn.ipynb
        - room-sensing.ipynb
        - room-sensing-feature-extraction-paper-implementations.ipynb
        - gan-based-room-imaging.ipynb
    - README.md
```

---

# Background

## What is WiFi CSI?

WiFi devices using IEEE 802.11n/ac transmit data using **Orthogonal Frequency Division Multiplexing (OFDM)**. Rather than measuring only the received signal strength, CSI measures the wireless channel response for every individual subcarrier.

For each subcarrier,

[
H(f)=Ae^{j\phi}
]

where

* **A** represents signal amplitude.
* **shi** represents phase.

These values change whenever the propagation environment changes.

Examples include

* human motion,
* furniture movement,
* wall reflections,
* drone movement,
* antenna orientation.

Because CSI preserves this detailed information, it contains significantly more spatial information than RSSI.

---

# Literature Review

## SpotFi (SIGCOMM 2015)

SpotFi demonstrated that commodity WiFi cards could achieve sub-meter indoor localization without specialized hardware.

Instead of estimating only the Angle of Arrival (AoA), SpotFi jointly estimates

* Angle of Arrival (AoA)
* Time of Flight (ToF)

using CSI collected across multiple antennas and OFDM subcarriers.

One important observation from SpotFi is that **CSI amplitude alone is insufficient**. The phase relationship across subcarriers also contains valuable localization information.

### What I learned

While implementing my parser, I noticed why SpotFi places significant emphasis on CSI preprocessing. Raw CSI packets contain noisy amplitude and phase measurements, making preprocessing an essential step before feature extraction.

The paper also motivated my use of

* CSI amplitude visualization
* FFT analysis
* Channel response analysis

inside the feature extraction notebook.

---

## DeepFi (WCNC 2015)

DeepFi treats CSI measurements as fingerprints.

Instead of manually engineering features, it learns compact latent representations using deep neural networks.

Fingerprinting consists of

Training Phase

CSI -> Deep Network -> Location Database

Testing Phase

CSI -> Deep Network -> Predicted Location

### What I learned

DeepFi demonstrates that dimensionality reduction is extremely important.

Inspired by this paper, I implemented

* FFT feature extraction
* Principal Component Analysis (PCA)
* Random Forest comparison

to investigate whether reducing feature dimensionality improves classification performance.

Although my implementation does not reproduce DeepFi exactly, it follows the same design philosophy of transforming high-dimensional CSI into compact discriminative features.

---

## WiFi Sensing Survey

The survey provides one of the most comprehensive overviews of modern CSI sensing.

The general sensing pipeline is

Raw CSI
    |
Preprocessing
    |
Noise Removal
    |
Feature Extraction
    |
Machine Learning

This survey strongly influenced the overall organization of my implementation.


---

## Wi-Drone

Wi-Drone extends CSI sensing toward autonomous drones.

Rather than estimating only position, Wi-Drone estimates

* Position
* Roll
* Pitch
* Yaw

using WiFi CSI together with sensor fusion.

The paper demonstrates that CSI is capable of supporting drone localization, although practical deployment remains challenging.

### What I learned

This paper showed that CSI alone is unlikely to replace traditional drone sensors.

Instead, CSI should be viewed as an additional sensing modality that complements

* cameras,
* IMUs,
* SLAM,
* visual odometry.

---

## Wi-Vi

Wi-Vi demonstrates that WiFi signals can detect motion even through walls.

Although the paper focuses on through-wall sensing rather than localization, it highlights the remarkable amount of environmental information contained within WiFi signals.

This inspired my exploration of CSI-based room imaging using generative models.

---

# Implementation

Rather than reproducing a single research paper, I implemented several components from different papers to understand the complete CSI sensing pipeline.

---

# 1. Room Sensing

The notebook **room-sensing.ipynb** implements an end-to-end CSI room classification pipeline.

## Data Loading

The Intel 5300 CSI Tool is used through the `csiread` library.

Each `.dat` file contains raw CSI packets collected from multiple antennas.

The filename also encodes metadata including

* user
* room location
* orientation
* activity
* trial
* repetition

which is automatically parsed into labels.

---

## CSI Preprocessing

Several preprocessing operations are applied before training.

### Hann Window

Each CSI packet is multiplied by a Hann window.

This reduces spectral leakage before transforming the signal into the delay domain.

### Channel Impulse Response

Instead of using raw CSI directly, the notebook converts CSI into a **Channel Impulse Response (CIR)** using the Inverse Fast Fourier Transform (IFFT).

The CIR provides a representation of signal propagation delays, making multipath reflections easier to analyze.

### Packet Resampling

Different recordings contain different numbers of packets.

To produce fixed-size inputs, every recording is resampled to **1024 packets**.

### Normalization

Each sample is normalized to the range

0 -> 1

ensuring stable CNN training.

---

## CNN Architecture

The classifier consists of

* 4 convolutional layers
* Batch Normalization
* Max Pooling
* Global Average Pooling
* Dense layers
* Dropout
* Softmax output

The network predicts one of eight room locations.

After training,

* training accuracy,
* validation accuracy,
* loss curves,
* confusion matrix

are generated.


---

# 2. Gesture Recognition

The notebook **gesture_cnn.ipynb** focuses on gesture classification using Doppler velocity spectra.

Unlike the room sensing notebook, this dataset is stored as MATLAB `.mat` files.

Each sample contains a three-dimensional tensor called

```
velocity_spectrum_ro
```

representing Doppler velocity over time.

---

## Data Preparation

Since gesture recordings contain different numbers of frames,

each recording is padded or truncated so every sample contains the same temporal length.

The data is then

* normalized,
* rearranged into TensorFlow format,
* split into training and testing sets.

---

## CNN-LSTM Model

The architecture combines

TimeDistributed CNN
    |
Flatten
    |
LSTM
    |
Dense Layers
    |
Gesture Prediction

The CNN extracts spatial Doppler features,

while the LSTM models temporal motion across consecutive frames.

The notebook reports

* accuracy,
* confusion matrix,
* classification report,

---

# 3. Feature Extraction

The notebook **feature_extraction_gesture.ipynb** investigates CSI visualization and handcrafted feature extraction.

Implemented analyses include

* heatmaps
* 3D surface plots
* temporal activity plots
* animations
* statistical descriptors

including

* mean
* standard deviation
* range
* median
* interquartile range (IQR)

This notebook helped me understand how different gestures produce distinct Doppler signatures before applying machine learning.

---

# 4. Paper Implementations

The notebook **room-sensing-feature-extraction-paper-implementations.ipynb** is the closest implementation to the research papers.

Instead of relying entirely on existing libraries, I manually parsed Intel CSI packets.

Implemented components include

* binary CSI parsing
* packet decoding
* CSI amplitude extraction
* phase extraction
* heatmap visualization
* FFT analysis
* PCA dimensionality reduction
* Random Forest classification

The FFT implementation was inspired by SpotFi,

while PCA follows ideas presented in DeepFi.

This notebook helped me understand how researchers transform raw CSI into meaningful machine learning features.

---

# 5. GAN-Based Room Imaging

As an extension beyond the assignment,

I investigated whether CSI could be used for room reconstruction.

The notebook

**gan-based-room-imaging.ipynb**

implements

* Butterworth filtering
* CSI normalization
* Conditional GAN
* Generator
* Discriminator
* Synthetic room-layout generation

Although this is an exploratory implementation rather than a reproduction of a published paper, it demonstrates a possible future research direction where CSI is used not only for localization but also for environmental reconstruction.

---

# Overall Processing Pipeline

Across all notebooks, the complete workflow is

```text
Raw CSI
    |
CSI Parsing
    |
Noise Reduction
    |
Hann Window
    |
IFFT
    |
Channel Impulse Response
    |
Normalization
    |
Feature Extraction
    |
CNN / CNN-LSTM / Random Forest / GAN
```

---

# What I Learned

Reading the papers and implementing the algorithms provided several practical insights.

### CSI preprocessing is critical

The quality of preprocessing has a significant impact on model performance. Small changes in filtering, normalization, or windowing noticeably affect the extracted features.

### Feature engineering remains valuable

Although deep learning automatically learns representations, handcrafted features such as amplitude statistics, FFT spectra, and PCA still provide useful information and improve understanding of the signal.

### Deep learning simplifies feature extraction

The CNN and CNN-LSTM models learn hierarchical representations directly from CSI data, reducing the need for manual engineering while achieving good classification performance.

### CSI contains rich environmental information

The experiments confirmed that CSI captures reflections, multipath propagation, and environmental structure well enough to distinguish rooms and gestures.

### Research papers focus heavily on preprocessing

One surprising observation was that much of the complexity lies in preprocessing rather than the neural network itself. Parsing binary CSI packets, calibrating signals, and reducing noise are essential before machine learning can be applied effectively.

---

# Challenges for Drone Navigation

Most publicly available CSI datasets assume a static transmitter and receiver.

A moving drone introduces additional challenges:

* continuously changing multipath
* changing antenna orientation
* motor vibrations
* Doppler shifts
* rapid channel variation
* synchronization issues

These factors mean that models trained on static datasets may not generalize well to real autonomous flight.

Future systems will likely combine CSI with

* IMU
* Visual SLAM
* Cameras
* LiDAR

instead of relying on CSI alone.

---

# Conclusion

This project provided hands-on experience with WiFi CSI sensing by combining theoretical understanding with practical implementation.

Rather than reproducing a single paper, I implemented techniques inspired by SpotFi, DeepFi, WiFi sensing surveys, and Wi-Drone to understand the complete CSI processing pipeline.

The experiments demonstrate that WiFi CSI contains sufficient information for localization, gesture recognition, and environmental sensing. At the same time, they highlight the importance of signal preprocessing, feature engineering, and robust machine learning models.

The project also reinforced that CSI has strong potential as a complementary sensing modality for autonomous drones operating in challenging indoor environments where traditional vision-based navigation is unreliable.

---

# What Changes When the Receiver is a Drone?

One of the key assumptions made by most WiFi CSI datasets, including Widar 3.0 and many of the datasets used in the literature, is that the receiver remains **stationary** while CSI measurements are collected. This assumption simplifies the wireless channel because changes in CSI are mainly caused by movement within the environment rather than movement of the receiver itself.

An autonomous drone violates this assumption completely. Unlike a fixed receiver, a drone is continuously moving, rotating, and vibrating, causing the wireless channel to change even if the surrounding environment remains unchanged. These additional sources of variation make CSI-based localization significantly more challenging.

## 1. Constant Motion

A drone is constantly changing its position while flying. Every small movement changes the propagation path between the transmitter and receiver.

As the drone moves, several channel characteristics change simultaneously:

* Direct path length
* Reflection path lengths
* Multipath propagation
* Signal attenuation
* Phase delay

Since CSI is extremely sensitive to these changes, the measured channel varies continuously instead of remaining stable.

For static localization systems, CSI fingerprints collected at one location remain relatively consistent. For a moving drone, however, the CSI fingerprint evolves continuously, making traditional fingerprint matching much less reliable.

---

## 2. Rotation and Orientation Changes

Unlike a stationary receiver, a drone continuously changes its orientation during flight.

Typical drone motion includes

* Roll
* Pitch
* Yaw

These rotations alter the orientation of the WiFi antennas relative to the access point.

Changing antenna orientation affects

* Antenna gain
* Signal polarization
* Received signal strength
* Phase measurements

Consequently, two identical positions may produce different CSI measurements simply because the drone is facing a different direction.

This introduces additional uncertainty that does not exist in static CSI datasets.

---

## 3. Motor Vibrations

Drone propellers generate continuous mechanical vibrations throughout the airframe.

These vibrations cause small but rapid movements of the antennas, introducing additional fluctuations into the measured wireless channel.

The resulting effects include

* Increased phase noise
* Small Doppler variations
* Reduced measurement stability
* Higher signal variance

Unlike laboratory CSI datasets collected using static tripods, real drones experience these vibrations throughout flight.

As a result, CSI measurements become considerably noisier.

---

## 4. Dynamic Multipath Propagation

Indoor environments naturally contain many reflected WiFi paths created by

* Walls
* Ceilings
* Floors
* Furniture
* Doors
* People

For a stationary receiver, these reflected paths remain relatively stable.

When the receiver is mounted on a drone, however, every movement changes

* Reflection angles
* Reflection distances
* Relative signal strengths
* Number of significant propagation paths

Objects that were previously dominant reflectors may disappear, while previously weak reflections may become dominant.

Therefore, the multipath channel evolves continuously throughout flight.

---

## 5. Doppler Shift

A moving receiver introduces Doppler shifts into the received signal.

The Doppler frequency depends on

* Flight speed
* Direction of motion
* Relative movement between transmitter, receiver, and surrounding objects

Most fingerprinting approaches assume Doppler effects are either negligible or caused primarily by human motion.

For an autonomous drone, Doppler becomes an inherent property of every CSI measurement.

If not properly compensated, these frequency shifts can reduce localization accuracy.

---

## 6. Flight Speed

CSI packets are collected sequentially over time.

When the receiver is stationary, consecutive packets correspond to nearly identical physical locations.

For a drone, this assumption is no longer valid.

At higher flight speeds,

* consecutive CSI packets represent different spatial positions,
* temporal alignment becomes more difficult,
* packet synchronization becomes increasingly important.

This creates additional challenges when constructing machine learning datasets because neighbouring samples may no longer represent the same location.

---

## 7. Environmental Dynamics

Real deployment environments are rarely static.

In addition to drone movement, the environment may also contain

* Moving people
* Opening or closing doors
* Moving vehicles
* Rotating machinery
* Other wireless devices

These environmental changes modify the wireless channel independently of the drone's motion, making it even harder to isolate location-specific CSI patterns.

A practical localization system must therefore distinguish between

* changes caused by drone motion,
* changes caused by the surrounding environment.

---

## 8. Implications for Machine Learning

Most CSI localization models, including fingerprinting methods such as DeepFi, are trained using measurements collected under relatively stable conditions.

When applied to a moving drone, several problems arise:

* The distribution of CSI measurements changes continuously.
* Previously learned fingerprints may no longer match.
* Models trained on static datasets may fail to generalize.
* Additional preprocessing and calibration become necessary.
* More robust temporal models are required to capture rapidly changing channel characteristics.

Cons

- **Angle of Arrival (AoA) continuously changes** as the drone moves and rotates, making AoA estimation much less stable than for a fixed receiver.

- **CSI fingerprints become invalid** because the receiver is constantly changing position, unlike static fingerprinting systems such as DeepFi.

- **Rapidly changing multipath propagation** causes the wireless channel to vary even when the environment remains unchanged.

- **Roll, pitch, and yaw rotations** alter antenna orientation, polarization, and received signal characteristics, producing CSI variations unrelated to location.

- **Propeller vibrations** introduce additional phase noise and signal instability, reducing the reliability of CSI measurements.

- **Doppler shifts are introduced by drone motion**, making it more difficult to distinguish between drone movement and moving objects in the environment.

- **Temporal synchronization becomes more challenging**, since consecutive CSI packets correspond to different physical locations during flight.

- **Environmental changes and drone motion occur simultaneously**, making it difficult to determine whether CSI variations are caused by the drone or by changes in the surroundings.

- **High computational complexity** of algorithms such as SpotFi (AoA/ToF estimation) may not be suitable for real-time execution on resource-constrained embedded flight controllers.

- **Frequent recalibration or online adaptation** may be required because CSI characteristics change with flight path, antenna placement, hardware differences, and environmental conditions.

- **Limited publicly available datasets** exist for moving drones, making it difficult to train and evaluate robust machine learning models.

- **CSI alone is unlikely to provide reliable navigation**, so practical systems must fuse CSI with IMU, cameras, LiDAR, or SLAM to achieve robust localization.

Pros

- **Works without GPS**, making it suitable for indoor environments where GPS signals are unavailable or unreliable.

- **Operates in low-light or dark environments**, unlike camera-based navigation systems that depend on good lighting conditions.

- **Can penetrate certain obstacles**, allowing sensing through smoke, dust, fog, and some non-metallic materials where optical sensors struggle.

- **Existing WiFi infrastructure can be reused**, reducing the need for additional localization hardware in many buildings.

- **Provides fine-grained channel information**, including amplitude and phase for each OFDM subcarrier, making it much richer than RSSI-based localization.

- **Sensitive to environmental changes**, enabling applications such as gesture recognition, human activity recognition, occupancy detection, and indoor localization.

- **Can complement existing drone sensors**, improving robustness when fused with IMU, cameras, LiDAR, or SLAM.

- **Supports device-free sensing**, allowing detection of people or movement without requiring users to carry any wireless device.

- **Potentially lower deployment cost**, since many indoor environments already contain WiFi access points.

- **Machine learning models can automatically learn CSI patterns**, enabling accurate classification of rooms, gestures, and activities.

- **Less affected by visual challenges** such as shadows, reflections, glare, or textureless surfaces that often reduce camera performance.

- **Provides an additional sensing modality**, increasing redundancy and improving reliability in autonomous navigation systems.

This explains why many current CSI localization systems achieve excellent accuracy in controlled laboratory environments but are considerably more difficult to deploy on autonomous aerial robots.


---

# References

1. Kotaru et al. *SpotFi: Decimeter Level Localization Using WiFi.*
2. Wang et al. *DeepFi: Deep Learning for Indoor Fingerprinting.*
3. Ma et al. *WiFi Sensing Survey.*
4. Chi et al. *Wi-Drone.*
5. Adib et al. *Wi-Vi: Through-Wall WiFi Sensing.*
