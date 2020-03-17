Camera Trace Erasing
====
Chang Chen, Zhiwei Xiong, Xiaoming Liu, Feng Wu. [Camera Trace Erasing](https://arxiv.org/pdf/2003.06951.pdf). In CVPR 2020. <br/>

## Requirements
Anaconda>=5.2.0 (Python 3.6) <br/>
PyTorch>=1.0.1 <br/>
Matlab R2018b (and above) <br/>

## Datasets and Results
Download \*.zip files, and unzip them to "datas" and "results", respectively <br/>
[http://pan.bitahub.com/index.php?mod=shares& <br/>
sid=eTJ2bFFQR3BzTm5FTGxONHJ3WXZzTTlobjItSTFzYl9vTmVySlE](http://pan.bitahub.com/index.php?mod=shares&sid=eTJ2bFFQR3BzTm5FTGxONHJ3WXZzTTlobjItSTFzYl9vTmVySlE)

## Descriptions
```
├─datas
│  ├─KCMI-550.zip
│  │  ├─KCMI-550                    // Images for validation
│  │  ├─KCMI-550-Val                // List of images
│  ├─VISION-1500.zip
│  │  ├──VISION-1500                // Images for validation
│  ├─KCMI+.zip
│  │  ├─KCMI+                       // Images for training
├─libs
│  ├─CameraFingerprint              // Fingerprint and PCE for verification
│  ├─KMCUDA                         // Source code for building
│  ├─libKMCUDA.so                   // Library for CUDA-based K-means
├─models                            // Pre-trained models
├─results
│  ├─KCMI-550-Ours.zip
│  │  ├──KCMI-550-Ours              // Generated results
│  │  ├──KCMI-550-Crop              // Centrally cropped images
│  │  ├──KCMI-550-Crop-Ours         // Generated results
│  │  ├──KCMI-Fingerprint           // Fingerprint of training data
│  ├─VISION-1500-Ours.zip
│  │  ├──VISION-1500-Ours           // Generated results
│  │  ├──VISION-1500-Ours-ResNet    // Extracted features of results
│  │  ├──VISION-1500-Ours-DenseNet  // Extracted features of results
├─inference_classification.py       // Classification task
├─inference_clustering.py           // Clustering task
├─inference_verification.m          // Verification task
├─inference_siamte.py               // Method for camera trace erasing
├─inference_feature.py              // Feature extraction for clustering
├─measure_l1.py                     // L1 distance
├─measure_niqe.m                    // Naturalness Image Quality Evaluator
```
