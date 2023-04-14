<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Virtual-Trial-Room</h3>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#dataset-and-preprocessing">Dataset and Preprocessing</a></li>
        <li><a href="#execution">Execution</a></li>
      </ul>
    </li>
    <li><a href="#results">Results</a></li>
    <li><a href="#contributions">Contributions</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

<img src="/assets/res1.png" width="800" height="500">

The fashion industry has always been a highly competitive and constantly evolving field. The traditional shopping experience involves trying on clothes in physical stores, but with the advent of online shopping, customers are unable to try on clothes before purchasing them. To address these challenges, there is a need for a virtual try-on solution that can accurately simulate the fit of clothing on a user.
In this project, we tackle the problem of simulating a virtual trial room. We are given an image of a person and a cloth as input pair and we need to generate an image of the person wearing the given cloth as output. We collect pair of person and cloth images from web as our dataset to increase diversity. This implementation is from: [Single Stage Virtual Try-on via Deformable Attention Flows](https://github.com/OFA-Sys/DAFlow).
For preprocessing, we also use implementations from [Self-Correction-Human-Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) and [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)<br/>
<!--For more details, please see [Report](/Report/)-->

<p align="right">(<a href="#top">back to top</a>)</p>

### Built With

* [PyTorch 1.13.1](https://pytorch.org/)
* [NumPy](https://numpy.org/)
* [Pillow](https://pillow.readthedocs.io/en/stable/installation.html)
* [OpenCV](https://opencv.org/)
* [Flask](https://flask.palletsprojects.com/en/2.2.x/)
* [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/03_python_api.md)
* [Mediapipe](https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/kanthprashant/Virtual-Trial-Room.git
   ```
2. Create a virtual environmnet using environment.yml
    ```sh
   conda env create -f environment.yml
   ```
3. Activate environment
    ```sh
   conda activate project 
   or
   source activate project
   ```

### Dataset and Preprocessing

<img src="/assets/preprocessing_1.png" width="700" height="500">

1. Use [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) for segmentation masks on lip weights and [get_img_agnostic.py](/input_formatter/get_img_agnostic.py) to obtain cloth agnostic images. Download their lip weights and put it under checkpoints folder, renamed as `lip.pth`.
2. Use [blazepose_keypoints.py](/input_formatter/blazepose_keypoints.py) and [draw_keypoints.py](/input_formatter/draw_keypoints.py) to obtain keypoints image.

<img src="/assets/preprocessing_2.png" width="700" height="500">

### Execution

Download weights from [openpose_finetune.pt](https://drive.google.com/drive/folders/1dZfctgSZA577MaCUo-sN78Cn4F6JKsF9?usp=share_link)

Put all model weights under checkpoints folder and check their names in VITON_Infer.py
1. Inference (single image)
```
python VITON_Infer.py --img_path <person_image_path> --cloth_path <cloth_image_path> --out_path <output_folder_path>
```
2. Run Flask server (default port 5000)
```
python app.py <path_of_folder_containing_clothes_images>
```
3. Train
```
python train_SDAFNet.py --name <name_to_identify_training> 
```
*Few points to help during training, the network is sensitive to learning rate and weight initialisation. Default weight initialisation seems to work better for training from scratch.*</br>
*There might be issues like white output being generated, if you keep learning rate too high or if weight initialisation is not proper.*

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- RESULTS -->
## Results
<img src="/assets/1.jpg" width="700" height="500">
<img src="/assets/2.png" width="700" height="500">

<!-- CONTRIBUTIONS -->
## Contributors

1. [Prashant Kanth](https://github.com/kanthprashant)
2. [Parth Goel](https://github.com/parthgoe1)
3. [Aditya Bhat](https://github.com/adityacbhat)
4. [Rishika Bhanushali](https://github.com/rb-rishika)
5. [Sharath Punna](https://github.com/sharath-cgm)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
## License

The use of this code is RESTRICTED to non-commercial research and educational purposes as per [DAFlow](https://github.com/OFA-Sys/DAFlow).

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* Bai, Shuai, et al. “Single Stage Virtual Try-on via Deformable Attention Flows.” ArXiv:2207.09161 [Cs], 19 July 2022, arxiv.org/abs/2207.09161. Accessed 13 Apr. 2023.
* Cao, Zhe, et al. “OpenPose: Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields.” ArXiv:1812.08008 [Cs], 30 May 2019, arxiv.org/abs/1812.08008.
* Bazarevsky, Valentin, et al. “BlazePose: On-Device Real-Time Body Pose Tracking.” ArXiv:2006.10204 [Cs], 17 June 2020, arxiv.org/abs/2006.10204.
* Li, Peike, et al. “Self-Correction for Human Parsing.” ArXiv:1910.09777 [Cs, Eess], 22 Oct. 2019, arxiv.org/abs/1910.09777.
* Choi, Seunghwan, et al. “VITON-HD: High-Resolution Virtual Try-on via Misalignment-Aware Normalization.” ArXiv:2103.16874 [Cs], 10 Sept. 2021, arxiv.org/abs/2103.16874. Accessed 13 Apr. 2023.
* Heusel, Martin, et al. “GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.” ArXiv:1706.08500 [Cs, Stat], 12 Jan. 2018, arxiv.org/abs/1706.08500.

<p align="right">(<a href="#top">back to top</a>)</p>
