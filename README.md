# [ECCV 24] Boost Your NeRF: A Model-Agnostic Mixture of Experts Framework for High Quality and Efficient Rendering

Welcome to the **official code repository** for our paper _"Boost Your NeRF: A Model-Agnostic Mixture of Experts Framework for High Quality and Efficient Rendering"_, accepted at **ECCV 2024**. 🚀

Our framework offers **state-of-the-art performance** by combining multiple experts in a model-agnostic manner to enhance the quality and efficiency of NeRF-based rendering.

![Teaser Image 1](method.png)  
<!-- ![Teaser Image 2](path_to_image2) -->

---

## 🔗 Project Page
Explore the full project details and supplementary materials on our [**Project Page**](https://eidoslab.github.io/boost-your-nerf/).

---


## 📊 Datasets
- Tested datasets:
    - *Bounded inward-facing*: [NeRF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1), [NSVF](https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NSVF.zip), [T&T (masked)](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip).
    - *Foward-facing*: [LLFF](https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7).

---
## 🚀 Quickstart

We offer our implementation with 3 main Fast-NeRFs architectures: **DVGO**, **TensoRF** and **Instant-NGP**. So far the code is available for DVGO: we will update our repo soon with the other method. 

Our method works under the same configuration of the original implementation of [**DVGO**](https://github.com/sunset1995/DirectVoxGO). So, the first step is to clone from thre and install all the dependencies:


```bash
git clone git@github.com:sunset1995/DirectVoxGO.git
cd DirectVoxGO
pip install -r requirements.txt
```

Then, clone our repository and **copy** the following files into the root directory of DVGO:

- `train_single_model.py`: A wrapper to the original DVGO method to train multiple models at different resolutions.
- `run_dvgo.py`: The main DVGO file with minor changes.
- `moe.py`: Contains all the logic behind our Sparse-Mixture of Experts framework.
- `moe_main.py`: The main file of our method.

The first step is to pre-train models at different resolutions:


```bash
python3 train_single_model.py --dataset_name nerf --scene lego --resolutions 128 160 200 256 300 350 --datadir path_to_datadir --render_test --eval_ssim --eval_lpips_alex
```

We are now ready to ready to train our MoE:


```bash
python3 moe_main.py --dataset_name nerf --scene lego --resolutions 128 160 200 256 300 350 --datadir path_to_datadir --render_test --eval_ssim --eval_lpips_alex --top_k 2 --num_experts 5
```

This code trains our MoE with *five experts* (the first model with a res of 128^3 is excluded as it is used for initializing the gate). 