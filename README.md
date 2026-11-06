# Real-Time-Kitchen-Hygiene-and-Safety-Monitoring-Using-YOLO11-and-TensorRT-on-NVIDIA-Jetson-Nano
# Real-Time Kitchen Hygiene and Safety Monitoring

**YOLO11s · TensorRT FP16 · NVIDIA Jetson Orin Nano**

A real-time, edge-AI system that monitors kitchen hygiene through a USB
webcam. It detects required protective equipment (PPE), identifies
missing PPE, and detects cockroaches. A live dashboard displays
detections, violation counts, FPS, and timestamped alerts.

The system performs inference locally on an NVIDIA Jetson Orin Nano,
without requiring a separate cloud server.

## Overview

Routine kitchen inspections happen periodically, which can allow hygiene
violations to go unnoticed between checks. This project uses computer
vision to continuously monitor a camera feed and flag detected
violations.

The detection model is **YOLO11s**, trained on a merged kitchen-hygiene
and cockroach dataset. The trained model is exported to ONNX and
converted into an FP16 TensorRT engine for deployment on the Jetson Orin
Nano.

### What it detects

-   Mask
-   Gloves
-   Hairnet
-   Missing mask (`without_mask`)
-   Missing gloves (`without_gloves`)
-   Missing hairnet (`without_hairnet`)
-   Cockroach

**Violation rule:** a detection of `without_mask`, `without_gloves`,
`without_hairnet`, or `cockroach` is counted as a hygiene violation.

## Key Results

  Metric                                                  Result
  -------------------------------------- -----------------------
  Model                                                  YOLO11s
  Model parameters                                   9.4 million
  Training duration                                   150 epochs
  Input resolution                                     640 × 640
  Validation images                                        2,351
  Validation instances                                     3,936
  Precision                                                0.965
  Recall                                                   0.957
  mAP@50                                                   0.966
  mAP@50:95                                                0.768
  F1-score                                                 0.961
  Live camera / inference speed                     About 15 FPS
  Dashboard display speed                                 30 FPS
  Observed per-frame dashboard latency                 35--43 ms
  FP16 TensorRT engine size                Approximately 17.9 MB

The validation metrics are from the trained model evaluated on the
validation set. Live performance was measured on the Jetson Orin Nano
deployment.

## System Architecture

### Training pipeline

1.  Combine the kitchen-hygiene and cockroach datasets.
2.  Clean and merge annotations.
3.  Resize images to 640 × 640 and apply augmentation.
4.  Train YOLO11s for 150 epochs on an NVIDIA DGX B200.
5.  Save the trained weights (`best.pt`).
6.  Export the model to ONNX using opset 12 with graph simplification.

### Edge deployment pipeline

1.  Copy the ONNX model to the Jetson Orin Nano.
2.  Build an FP16 TensorRT engine on the Jetson using `trtexec`.
3.  Capture frames from a USB webcam using OpenCV.
4.  Run inference using the TensorRT engine.
5.  Post-process detections and apply the violation rule.
6.  Display the live camera feed and dashboard with class counts and
    timestamped alerts.

> **Important:** Build the TensorRT engine on the target Jetson.
> TensorRT engines are tied to the GPU and TensorRT environment for
> which they are built.

## Hardware Requirements

  Component     Specification
  ------------- -----------------------------------------------------
  Edge device   NVIDIA Jetson Orin Nano Developer Kit (Super)
  CPU           6-core 64-bit Arm Cortex-A78AE
  GPU           NVIDIA Ampere, 1,024 CUDA cores and 32 Tensor cores
  Memory        8 GB LPDDR5
  Camera        USB webcam
  Display       Monitor for the live dashboard
  Storage       microSD card or external NVMe SSD

The reported board specification lists AI performance up to 67 TOPS and
a 7--25 W power range.

### Training hardware

-   NVIDIA DGX B200 GPU server (RBU AI Center)

## Software Requirements

The deployment environment used in the project:

  Software / library   Version or configuration
  -------------------- ------------------------------------------------
  NVIDIA JetPack       6.2.1
  TensorRT             10.3.0 (provided with JetPack)
  OpenCV               4.10.0.84
  NumPy                1.26.4
  Python               Python 3 environment on Jetson
  CUDA                 CUDA version supplied by the installed JetPack

For training, the paper reports Python 3.12.3, PyTorch 2.8.0, and
Ultralytics 8.4.178.

**TensorRT compatibility note:** Do not install an unrelated TensorRT
version using `pip` on the Jetson. The project used TensorRT 10.3.0
supplied with JetPack 6.2.1. A separately installed TensorRT 11.4 build
targeting CUDA 13 caused a CUDA runtime mismatch on the Jetson
environment.

## Installation and Setup

The exact filenames and directory structure may vary depending on the
repository version. The steps below describe the deployment workflow
used by the project.

### 1. Prepare the Jetson

Install and configure NVIDIA JetPack 6.2.1 on the Jetson Orin Nano.
Confirm that the CUDA and TensorRT components supplied with JetPack are
available.

### 2. Create a Python environment

Because the deployment uses the system TensorRT package, create the
virtual environment with access to system site packages:

``` bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
```

### 3. Install Python dependencies

Install a compatible OpenCV and NumPy environment:

``` bash
pip install numpy==1.26.4
pip install opencv-python==4.10.0.84
```

TensorRT 10.3.0 should be available from the JetPack installation.
Verify that it can be imported:

``` bash
python3 -c "import tensorrt as trt; print(trt.__version__)"
```

Do not use `pip` to replace the JetPack-provided TensorRT installation.

### 4. Export the trained model to ONNX

On the training machine, export the trained YOLO11s weights:

``` bash
yolo export model=best.pt format=onnx opset=12 simplify=True
```

Copy the resulting ONNX file to the Jetson.

### 5. Build the TensorRT FP16 engine

On the Jetson, use the TensorRT `trtexec` utility:

``` bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=best.onnx \
  --saveEngine=model_fp16.engine \
  --fp16
```

The resulting engine in the reported setup was approximately 17.9 MB.
The engine must be built on the target Jetson with its installed
TensorRT version.

### 6. Run the application

Use the repository's dashboard-enabled webcam inference script after
placing the TensorRT engine and class-label file at the paths expected
by that script.

``` bash
python3 <dashboard_inference_script>.py
```

Replace the placeholder with the actual script filename in the
repository. Connect the USB webcam and monitor before starting the
application.

## Dashboard

The dashboard provides:

-   Live webcam feed with detection bounding boxes
-   Class-wise detection counts
-   Violation warning banner and total violation count
-   Camera, inference, and display FPS
-   Recent alerts with timestamps

Green boxes indicate required items detected as present. Red boxes
indicate detected violations.

## Dataset

The model was trained using two public datasets from Roboflow Universe:

1.  **Kitchen Hygiene Monitor** --- [Roboflow Universe
    dataset](https://universe.roboflow.com/gunjans-workspace-zqnbp/kitchen-hygiene-monitor-hlecs/dataset/1)
2.  **Cockroach** --- [Roboflow Universe
    dataset](https://universe.roboflow.com/safety-food-system-39okc/cockroach-gkzut-sb3x5/dataset/1)

Both datasets are identified in the paper as licensed under **CC BY
4.0**. Review the original dataset pages and their attribution
requirements before redistribution or reuse.

The merged dataset uses the following labels:

``` text
cockroach
gloves
mask
hairnet
without_gloves
without_hairnet
without_mask
unused_1
unused_6
```

The `unused_1` and `unused_6` labels are identified as background
classes in the paper's validation results.

## Training Configuration

  Parameter           Value
  ------------------- --------------------------
  Model               YOLO11s
  Image size          640 × 640
  Epochs              150
  Batch size          16
  Workers             8
  Training hardware   NVIDIA DGX B200
  Ultralytics         8.4.178
  PyTorch             2.8.0
  Python              3.12.3
  Optimizer           Auto (Adam, as reported)

The training used the default Ultralytics augmentation pipeline,
including HSV colour jitter, random translation and scaling, and
horizontal flipping. Mosaic augmentation was disabled for the final 10
epochs.

## Deployment Performance

Measured on the Jetson Orin Nano with the dashboard-enabled inference
script:

  Metric                               Observation
  ----------------------------- ------------------
  Camera FPS                                    15
  Inference FPS                                 15
  Dashboard FPS                                 30
  Dashboard latency per frame            35--43 ms
  Pre-processing                            8.4 ms
  TensorRT inference                       24.7 ms
  Post-processing                           0.7 ms
  GPU load                                   44.2%
  CPU load per core                    17.0--45.9%
  Memory used                     2.6 GB of 7.4 GB
  Power mode                            MAXN_SUPER

Stage timings are measurements reported for the tested deployment and
may vary with the camera, scene, and device configuration.

## Limitations and Future Work

The project identified the following limitations:

-   `without_hairnet` was the weakest PPE violation class, with an
    mAP@50 of 0.870.
-   The cockroach was detected with low confidence (0.29) in one live
    test, despite a validation mAP@50 of 0.940. More cockroach images
    from kitchen-like environments are needed.
-   The system needs evaluation in a working kitchen to measure its
    false-alarm rate.

Planned future improvements include:

-   Add apron and rodent detection.
-   Collect more cockroach images from real kitchens.
-   Confirm violations across multiple consecutive frames or use object
    tracking to reduce false alerts.
-   Explore INT8 quantization for higher FPS.
-   Send alerts remotely to kitchen managers.
-   Conduct trials in an operational kitchen.

## Practical Troubleshooting

### NumPy / PyTorch compatibility

The training environment encountered a NumPy 2.x and PyTorch build
compatibility issue. The reported fix was to use NumPy 1.26.4, replace
the incompatible OpenCV 5.x package with a compatible OpenCV 4.x
headless build in the training container, restart the kernel, and
install Ultralytics without allowing it to upgrade dependencies.

### TensorRT import or CUDA mismatch on Jetson

The Jetson deployment encountered an issue when a separately installed
TensorRT 11.4 package targeted CUDA 13, while the JetPack 6 environment
used CUDA 12. The reported solution was to use TensorRT 10.3.0 supplied
with JetPack 6.2.1 and create the Python virtual environment with
`--system-site-packages`.

## Repository Contents

The paper describes the release of the following project assets:

-   Inference scripts for video, webcam, and dashboard use
-   Trained weights (`best.pt`)
-   ONNX model (`best.onnx`)
-   Class-label file

Refer to the repository for the actual filenames and current
availability of these assets.

## Authors

Department of Electronics Engineering, Ramdeobaba University, Nagpur,
India.

-   Rudram Dindorkar
-   Nishika Deotale
-   Namrata Thorat
-   Revanshu Pusadkar
-   Snehal Laddha

## Acknowledgement

The authors sincerely thank the RBU AI Center, Ramdeobaba University, for providing access to the NVIDIA DGX B200, which was utilized for model training. The authors also express their heartfelt gratitude to their project guide for valuable guidance, continuous support, and constructive suggestions throughout the course of this work.

## Publication

Project details and reported results are documented in the paper:

**"Real-Time Kitchen Hygiene and Safety Monitoring Using YOLO11 and
TensorRT on NVIDIA Jetson Orin Nano."**

## License

The two source datasets are identified as CC BY 4.0 in the paper. This
does not, by itself, establish the license for this project's source
code, trained weights, or other repository assets. Add an explicit
repository license if you intend to distribute those materials.

## Disclaimer

This project is an AI-assisted monitoring prototype. Detection results
can be affected by camera placement, lighting, occlusion, distance, and
differences between deployment scenes and training data. It should
support---not replace---regular hygiene inspections and applicable
food-safety procedures.
