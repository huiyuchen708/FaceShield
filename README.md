# FaceShield: High-Utility Differential Privacy Facial Protection for Online Image Publishing

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-brightgreen.svg)](https://creativecommons.org/licenses/by-nc/4.0/) ![PyTorch](https://img.shields.io/badge/Pytorch-2.6.0-brightgreen) ![Python](https://img.shields.io/badge/Python-3.10.16-brightgreen)

<b> Authors: Ming Zhang, Yuchen Hui, <a href='https://scholar.google.com/citations?hl=zh-CN&user=y52WOmkAAAAJ&view_op=list_works&sortby=pubdate'>Xiaoguang Li*</a>, Guojia Yu, <a href='https://scholar.google.com/citations?hl=zh-CN&user=FFX0Mj4AAAAJ'>Haonan Yan</a>, <a href='https://scholar.google.com/citations?hl=zh-CN&user=oEcRS84AAAAJ&view_op=list_works&sortby=pubdate'>Hui Li*</a>  </b>

---
## 📚 **Overview**
<div align="center"> 
</div>
<div style="text-align:center;">
  <img src="figures/framework.png" style="max-width:60%;">
</div>

Welcome to FaceShield, your high-fidelity and provable facial privacy protection platform! Here are some key features of our platform:

> 📌 **Feature disentanglement**: *Faceshield* leverages joint statistical and spatial independence constraints to maximally disentangle facial identity and perceptual content, thereby separating identity perturbation from perceptual preservation.
> 
> 📌 **Practical Sampling**: *Faceshield* decomposes high-dimensional identity perturbation into two tractable sub-processes, thereby reducing computational overhead in high-dimensional scenarios.
> 
> 📌 **Comprehensive Protection**: *Faceshield* prevents unintended identity leakage arising from the perceptual latent space, offering users comprehensive facial privacy protection.
> 
> 📌 **Extensive Evaluation**: *Faceshield* offers users a superior and reliable solution in terms of naturalness, perceptual fidelity, privacy protection, and efficiency.

---

## 😊 **FaceShield Updates**
> [x] To be supplemented after acceptance...
> ✅ 20/09/2025: *First version pre-released for this open source code.*
---

<font size=4><b> Table of Contents </b></font>

- [Quick Start](#-quick-start)
  - [Environmental](#1-Environmental)
  - [Download Data](#2-download-data)
  - [Preprocessing](#3-preprocessing)
  - [Training](#4-Training)
  - [Evaluation](#5-evaluation)
- [Results](#-results)
- [Citation](#-citation)
- [Copyright](#%EF%B8%8F-license)

---

## ⏳ Quick Start

### 1. Environmental
You can run the following script to configure the necessary environment:

```
git clone git@github.com:huiyuchen708/FaceShield.git
cd FaceShield
conda create -n FaceShield python=3.10.16
conda activate FaceShield
pip install -r requirements.txt
```

### 2. Download Data

<a href="#top">[Back to top]</a>

For convenience, we provide the data used in our study, including:
1. Downloading [Celeb-DF-v2](https://github.com/ondyari/FaceForensics) *Original* dataset for training data preparation. All the downloaded datasets are already **preprocessed** to cropped faces (32 frames per video) with their masks and landmarks, which can be **directly deployed to training our benchmark**. For the test set, we provide a download link for [UADFV](https://github.com/ondyari/FaceForensics), while the other datasets can be obtained online.


🛡️ **Copyright of the above datasets belongs to their original providers.**

2. Upon downloading the datasets, please ensure to store them in the [`./content`](./content/) folder, arranging them in accordance with the directory structure outlined below:

```
content
├── Celeb-DF-v2
|   ├── Celeb-real
|   |   ├── frames (if you download my processed data)
|   |   |   ├── id*_*
|   |   |   |   ├── *.png
│   |   │   ├── ...
|   |   ├── landmarks (if you download my processed data)
|   |   |   ├── id*_*
|   |   |   |   ├── *.npy
│   |   │   ├── ...
|   ├── Celeb-synthesis
|   |   ├── frames (if you download my processed data)
|   |   |   ├── id*_id*_*
|   |   |   |   ├── *.png
│   |   │   ├── ...
|   |   ├── landmarks (if you download my processed data)
|   |   |   ├── id*_*
|   |   |   |   ├── *.npy
│   |   │   ├── ...
|   ├── Youtube-real
|   |   ├── frames (if you download my processed data)
|   |   |   ├── *
|   |   |   |   ├── *.png
│   |   │   ├── ...
|   |   ├── landmarks (if you download my processed data)
|   |   |   ├── *
|   |   |   |   ├── *.npy
│   |   │   ├── ...
|   ├── List_of_testing_videos.txt
├── UADFV
|   ├── fake
|   |   ├── frames (if you download my processed data)
|   |   |   ├── *_fake
|   |   |   |   ├── *.png
|   |   ├── landmarks (if you download my processed data)
|   |   |   ├── *_fake
|   |   |   |   ├── *.png
|   ├── real
|   |   ├── frames (if you download my processed data)
|   |   |   ├── *
|   |   |   |   ├── *.png
|   |   ├── landmarks (if you download my processed data)
|   |   |   ├── *
|   |   |   |   ├── *.npy
Other datasets are similar to the above structure
```

If you choose to store your datasets in a different folder, you may specified the `data_root` in `train_Reconstructor.py` and `train_SeparationExtractor.py`.

### 3. Preprocessing

<a href="#top">[Back to top]</a>

**❗️Note**: If you want to directly utilize the data, including frames, landmarks, masks, and more, that I have provided above, you can skip the pre-processing step. **However, you still need to run the rearrangement script to generate the JSON file** for each dataset for the unified data loading in the training and testing process.

DeepfakeBench follows a sequential workflow for face detection, alignment, and cropping. The processed data, including face images, landmarks, and masks, are saved in separate folders for further analysis.

To start preprocessing your dataset, please follow these steps:

1. Download the [shape_predictor_81_face_landmarks.dat](https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/shape_predictor_81_face_landmarks.dat) file. Then, copy the downloaded shape_predictor_81_face_landmarks.dat file into the `./preprocessing/dlib_tools folder`. This file is necessary for Dlib's face detection functionality.

2. Open the [`./preprocessing/config.yaml`](./preprocessing/config.yaml) and locate the line `default: DATASET_YOU_SPECIFY`. Replace `DATASET_YOU_SPECIFY` with the name of the dataset you want to preprocess, such as `FaceForensics++`.

7. Specify the `dataset_root_path` in the config.yaml file. Search for the line that mentions dataset_root_path. By default, it looks like this: ``dataset_root_path: ./datasets``.
Replace `./datasets` with the actual path to the folder where your dataset is arranged. 

Once you have completed these steps, you can proceed with running the following line to do the preprocessing:

```
cd preprocessing

python preprocess.py
```
You may skip the preprocessing step by downloading the provided data.

### 4. Training
To simplify the handling of different datasets, we propose a unified and convenient way to load them. The function eliminates the need to write separate input/output (I/O) code for each dataset, reducing duplication of effort and easing data management.

After the preprocessing above, you will obtain the processed data (*i.e., frames, landmarks, and masks*) for each dataset you specify. Similarly, you need to set the parameters in `./preprocessing/config.yaml` for each dataset. After that, run the following line:
```
cd preprocessing

python rearrange.py
```
After running the above line, you will obtain the JSON files for each dataset in the `./preprocessing/dataset_json` folder. The rearranged structure organizes the data in a hierarchical manner, grouping videos based on their labels and data splits (*i.e.,* train, test, validation). Each video is represented as a dictionary entry containing relevant metadata, including file paths, labels, compression levels (if applicable), *etc*. 

<a href="#top">[Back to top]</a>

To run the training code, you should first download the pretrained weights for the corresponding **backbones** (These pre-trained weights are from ImageNet). You can download them from [Link](https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/pretrained.zip). After downloading, you need to put all the weights files into the folder `./training/pretrained`.

Then, you should go to the `./training/config/detector/` folder and then Choose the detector to be trained. For instance, you can adjust the parameters in [`xception.yaml`](./training/config/detector/xception.yaml) to specify the parameters, *e.g.,* training and testing datasets, epoch, frame_num, *etc*.

After setting the parameters, you can run with the following to train the Xception detector:

```
python training/train.py \
--detector_path ./training/config/detector/xception.yaml
```

You can also adjust the training and testing datasets using the command line, for example:

```
python training/train.py \
--detector_path ./training/config/detector/xception.yaml  \
--train_dataset "FaceForensics++" \
--test_dataset "Celeb-DF-v1" "Celeb-DF-v2"
```

By default, the checkpoints and features will be saved during the training process. If you do not want to save them, run with the following:

```
python training/train.py \
--detector_path ./training/config/detector/xception.yaml \
--train_dataset "FaceForensics++" \
--test_dataset "Celeb-DF-v1" "Celeb-DF-v2" \
--no-save_ckpt \
--no-save_feat
```

For **multi-gpus training** (DDP), please refer to [`train.sh`](./train.sh) file for details.

To train other detectors using the code mentioned above, you can specify the config file accordingly. However, for the Face X-ray detector, an additional step is required before training. To save training time, a pickle file is generated to store the Top-N nearest images for each given image. To generate this file, you should run the [`generate_xray_nearest.py`](./training/dataset/generate_xray_nearest.py) file. Once the pickle file is created, you can train the Face X-ray detector using the same way above. If you want to check/use the files I have already generated, please refer to the [`link`](https://github.com/SCLBD/DeepfakeBench/releases/tag/v1.0.2).


### 5. Evaluation
If you only want to evaluate the detectors to produce the results of the cross-dataset evaluation, you can use the the [`test.py`](./training/test.py) code for evaluation. Here is an example:

```
python3 training/test.py \
--detector_path ./training/config/detector/xception.yaml \
--test_dataset "Celeb-DF-v1" "Celeb-DF-v2" "DFDCP" \
--weights_path ./training/weights/xception_best.pth
```
**Note that we have provided the pre-trained weights for each detector (you can download them from the [`link`](https://github.com/SCLBD/DeepfakeBench/releases/tag/v1.0.1)).** Make sure to put these weights in the `./training/weights` folder.



## 🏆 Results

<a href="#top">[Back to top]</a>

>  ❗️❗️❗️ **DeepfakeBench-v2 Updates:**
> 
> The below results are cited from our [paper](https://arxiv.org/abs/2307.01426). We have conducted more comprehensive evaluations using the DeepfakeBench-v2, with more datasets used and more detectors implemented. We will update the below table soon.
> 

In our Benchmark, we apply [TensorBoard](https://github.com/tensorflow/tensorboard) to monitor the progress of training models. It provides a visual representation of the training process, allowing users to examine training results conveniently.

To demonstrate the effectiveness of different detectors, we present **partial results** from both within-domain and cross-domain evaluations. The evaluation metric used is the frame-level Area Under the Curve (AUC). In this particular scenario, we train the detectors on the FF++ (c23) dataset and assess their performance on other datasets.

For a comprehensive overview of the results, we strongly recommend referring to our [paper](https://arxiv.org/abs/2307.01426). These resources provide a detailed analysis of the training outcomes and offer a deeper understanding of the methodology and findings.


| Type     | Detector   | Backbone  | FF++\_c23 | FF++\_c40 | FF-DF   | FF-F2F  | FF-FS   | FF-NT   | Avg.     | Top3 | CDFv1   | CDFv2   | DF-1.0  | DFD     | DFDC    | DFDCP   | Fsh     | UADFV   | Avg.    | Top3 |
|----------|------------|-----------|------------|------------|---------|---------|---------|---------|----------|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|------|
| Naive    | Meso4      | MesoNet   | 0.6077     | 0.5920     | 0.6771  | 0.6170  | 0.5946  | 0.5701  | 0.6097   | 0    | 0.7358  | 0.6091  | 0.9113  | 0.5481  | 0.5560  | 0.5994  | 0.5660  | 0.7150  | 0.6551 | 1    |
| Naive    | MesoIncep  | MesoNet   | 0.7583     | 0.7278     | 0.8542  | 0.8087  | 0.7421  | 0.6517  | 0.7571   | 0    | 0.7366  | 0.6966  | 0.9233  | 0.6069  | 0.6226  | 0.7561  | 0.6438  | 0.9049  | 0.7364 | 3    |
| Naive    | CNN-Aug    | ResNet    | 0.8493     | 0.7846     | 0.9048  | 0.8788  | 0.9026  | 0.7313  | 0.8419   | 0    | 0.7420  | 0.7027  | 0.7993  | 0.6464  | 0.6361  | 0.6170  | 0.5985  | 0.8739  | 0.7020 | 0    |
| Naive    | Xception   | Xception  | 0.9637     | 0.8261     | 0.9799  | 0.9785  | 0.9833  | 0.9385  | 0.9450   | 4    | 0.7794  | 0.7365  | 0.8341  | **0.8163**  | 0.7077  | 0.7374  | 0.6249  | 0.9379  | 0.7718 | 2    |
| Naive    | EfficientB4| Efficient | 0.9567     | 0.8150     | 0.9757  | 0.9758  | 0.9797  | 0.9308  | 0.9389   | 0    | 0.7909  | 0.7487  | 0.8330  | 0.8148  | 0.6955  | 0.7283  | 0.6162  | 0.9472  | 0.7718 | 3    |
| Spatial  | Capsule    | Capsule   | 0.8421     | 0.7040     | 0.8669  | 0.8634  | 0.8734  | 0.7804  | 0.8217   | 0    | 0.7909  | 0.7472  | 0.9107  | 0.6841  | 0.6465  | 0.6568  | 0.6465  | 0.9078  | 0.7488 | 2    |
| Spatial  | FWA        | Xception  | 0.8765     | 0.7357     | 0.9210  | 0.9000  | 0.8843  | 0.8120  | 0.8549   | 0    | 0.7897  | 0.6680  | **0.9334**  | 0.7403  | 0.6132  | 0.6375  | 0.5551  | 0.8539  | 0.7239 | 1    |
| Spatial  | Face X-ray      | HRNet     | 0.9592     | 0.7925     | 0.9794  | **0.9872**  | 0.9871  | 0.9290  | 0.9391   | 3    | 0.7093  | 0.6786  | 0.5531  | 0.7655  | 0.6326  | 0.6942  | **0.6553**  | 0.8989  | 0.6985 | 0    |
| Spatial  | FFD        | Xception  | 0.9624     | 0.8237     | 0.9803  | 0.9784  | 0.9853  | 0.9306  | 0.9434   | 1    | 0.7840  | 0.7435  | 0.8609  | 0.8024  | 0.7029  | 0.7426  | 0.6056  | 0.9450  | 0.7733 | 1    |
| Spatial  | CORE       | Xception  | 0.9638     | 0.8194     | 0.9787  | 0.9803  | 0.9823  | 0.9339  | 0.9431   | 2    | 0.7798  | 0.7428  | 0.8475  | 0.8018  | 0.7049  | 0.7341  | 0.6032  | 0.9412  | 0.7694 | 0    |
| Spatial  | Recce      | Designed  | 0.9621     | 0.8190     | 0.9797  | 0.9779  | 0.9785  | 0.9357  | 0.9422   | 1    | 0.7677  | 0.7319  | 0.7985  | 0.8119  | 0.7133  | 0.7419  | 0.6095  | 0.9446  | 0.7649 | 2    |
| Spatial  | UCF        | Xception  | **0.9705** | **0.8399** | **0.9883** | 0.9840  | **0.9896** | **0.9441** | **0.9527** | **6** | 0.7793  | 0.7527  | 0.8241  | 0.8074  | **0.7191**  | **0.7594**  | 0.6462  | **0.9528**  | 0.7801 | **5** |
| Frequency| F3Net      | Xception  | 0.9635     | 0.8271     | 0.9793  | 0.9796  | 0.9844  | 0.9354  | 0.9449   | 1    | 0.7769  | 0.7352  | 0.8431  | 0.7975  | 0.7021  | 0.7354  | 0.5914  | 0.9347  | 0.7645 | 0    |
| Frequency| SPSL       | Xception  | 0.9610     | 0.8174     | 0.9781  | 0.9754  | 0.9829  | 0.9299  | 0.9408   | 0    | **0.8150**  | **0.7650**  | 0.8767  | 0.8122  | 0.7040  | 0.7408  | 0.6437  | 0.9424  | **0.7875** | 3    |
| Frequency| SRM        | Xception  | 0.9576     | 0.8114     | 0.9733  | 0.9696  | 0.9740  | 0.9295  | 0.9359   | 0    | 0.7926  | 0.7552  | 0.8638  | 0.8120  | 0.6995  | 0.7408  | 0.6014  | 0.9427  | 0.7760 | 2    |


In the above table, "Avg." donates the average AUC for within-domain and cross-domain evaluation, and the overall results. "Top3" represents the count of each method ranks within the top-3 across all testing datasets. The best-performing method for each column is highlighted.


Also, we provide all experimental results in [Link (code: qjpd)](https://pan.baidu.com/s/1Mgo5rW08B3ee_8ZBC3EXJA?pwd=qjpd). You can use these results for further analysis using the code in [`./analysis`](`./analysis`) folder.































## 📝 Citation

<a href="#top">[Back to top]</a>

If you find our benchmark useful to your research, please cite it as follows:

```
@inproceedings{DeepfakeBench_YAN_NEURIPS2023,
 author = {Yan, Zhiyuan and Zhang, Yong and Yuan, Xinhang and Lyu, Siwei and Wu, Baoyuan},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Neumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {4534--4565},
 publisher = {Curran Associates, Inc.},
 title = {DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/0e735e4b4f07de483cbe250130992726-Paper-Datasets_and_Benchmarks.pdf},
 volume = {36},
 year = {2023}
}
```

If interested, you can read our recent works about deepfake detection, and more works about trustworthy AI can be found [here](https://sites.google.com/site/baoyuanwu2015/home).
```
@inproceedings{UCF_YAN_ICCV2023,
 title={Ucf: Uncovering common features for generalizable deepfake detection},
 author={Yan, Zhiyuan and Zhang, Yong and Fan, Yanbo and Wu, Baoyuan},
 booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
 pages={22412--22423},
 year={2023}
}

@inproceedings{LSDA_YAN_CVPR2024,
  title={Transcending forgery specificity with latent space augmentation for generalizable deepfake detection},
  author={Yan, Zhiyuan and Luo, Yuhao and Lyu, Siwei and Liu, Qingshan and Wu, Baoyuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}

@inproceedings{cheng2024can,
  title={Can We Leave Deepfake Data Behind in Training Deepfake Detector?},
  author={Cheng, Jikang and Yan, Zhiyuan and Zhang, Ying and Luo, Yuhao and Wang, Zhongyuan and Li, Chen},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}

@article{chen2024textit,
  title={X^2-DFD: A framework for eXplainable and eXtendable Deepfake Detection},
  author={Chen, Yize and Yan, Zhiyuan and Lyu, Siwei and Wu, Baoyuan},
  journal={arXiv preprint arXiv:2410.06126},
  year={2024}
}

@article{cheng2024stacking,
  title={Stacking Brick by Brick: Aligned Feature Isolation for Incremental Face Forgery Detection},
  author={Cheng, Jikang and Yan, Zhiyuan and Zhang, Ying and Hao, Li and Ai, Jiaxin and Zou, Qin and Li, Chen and Wang, Zhongyuan},
  journal={arXiv preprint arXiv:2411.11396},
  year={2024}
}

@article{yan2024effort,
  title={Effort: Efficient Orthogonal Modeling for Generalizable AI-Generated Image Detection},
  author={Yan, Zhiyuan and Wang, Jiangming and Wang, Zhendong and Jin, Peng and Zhang, Ke-Yue and Chen, Shen and Yao, Taiping and Ding, Shouhong and Wu, Baoyuan and Yuan, Li},
  journal={arXiv preprint arXiv:2411.15633},
  year={2024}
}

```


## 🛡️ License

<a href="#top">[Back to top]</a>


This repository is licensed by [The Chinese University of Hong Kong, Shenzhen](https://www.cuhk.edu.cn/en) under Creative Commons Attribution-NonCommercial 4.0 International Public License (identified as [CC BY-NC-4.0 in SPDX](https://spdx.org/licenses/)). More details about the license could be found in [LICENSE](./LICENSE).

This project is built by the Secure Computing Lab of Big Data (SCLBD) at The School of Data Science (SDS) of The Chinese University of Hong Kong, Shenzhen, directed by Professor [Baoyuan Wu](https://sites.google.com/site/baoyuanwu2015/home). SCLBD focuses on the research of trustworthy AI, including backdoor learning, adversarial examples, federated learning, fairness, etc.

If you have any suggestions, comments, or wish to contribute code or propose methods, we warmly welcome your input. Please contact us at wubaoyuan@cuhk.edu.cn or yanzhiyuan1114@gmail.com. We look forward to collaborating with you in pushing the boundaries of deepfake detection.
