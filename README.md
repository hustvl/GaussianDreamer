![block](./images/title.gif)

# GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging 2D and 3D Diffusion Models (CVPR 2024)
### [Project Page](https://taoranyi.com/gaussiandreamer/) | [arxiv Paper](https://arxiv.org/abs/2310.08529)

[GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging 2D and 3D Diffusion Models](https://taoranyi.com/gaussiandreamer/)  

[Taoran Yi](https://github.com/taoranyi)<sup>1</sup>,
[Jiemin Fang](https://jaminfong.cn/)<sup>2‡</sup>, [Junjie Wang](https://scholar.google.com/citations?view_op=list_works&hl=zh-CN&user=9Nw_mKAAAAAJ)<sup>2</sup>, [Guanjun Wu](https://guanjunwu.github.io/)<sup>3</sup>,  [Lingxi Xie](http://lingxixie.com/)<sup>2</sup>, </br>[Xiaopeng Zhang](https://scholar.google.com/citations?user=Ud6aBAcAAAAJ&hl=zh-CN)<sup>2</sup>,[Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/)<sup>1</sup>, [Qi Tian](https://www.qitian1987.com/)<sup>2</sup> , [Xinggang Wang](https://xwcv.github.io/)<sup>1‡✉</sup>

<sup>1</sup>School of EIC, HUST &emsp;<sup>2</sup>Huawei Inc. &emsp; <sup>3</sup>School of CS, HUST &emsp; 

<sup>‡</sup>Project lead.  <sup>✉</sup>Corresponding author. 

![block](./images/teaser.png)
In recent times, the generation of 3D assets from text prompts has shown impressive results. Both 2D and 3D diffusion models can help generate decent 3D objects based on prompts. 3D diffusion models have good 3D consistency, but their quality and generalization are limited as trainable 3D data is expensive and hard to obtain. 2D diffusion models enjoy strong abilities of generalization and fine generation, but 3D consistency is hard to guarantee. This paper attempts to bridge the power from the two types of diffusion models via the recent explicit and efficient 3D Gaussian splatting representation. A fast 3D object generation framework, named as GaussianDreamer, is proposed, where the 3D diffusion model provides priors for initialization and the 2D diffusion model enriches the geometry and appearance. Operations of noisy point growing and color perturbation are introduced to enhance the initialized Gaussians. Our GaussianDreamer can generate a high-quality 3D instance or 3D avatar within 15 minutes on one GPU, much faster than previous methods, while the generated instances can be directly rendered in real time.
![block](./images/output_gs.gif)

## 🦾 Updates
- 6/26/2024: We have released [GaussianDreamerPro](https://taoranyi.com/gaussiandreamerpro/) with highly enhanced quality which can be seamlessly integrated into animation/simulation pipelines🚀.
- 5/14/2024: We update the results of our method on [T<sup>3</sup>Bench](https://t3bench.com/), refer to [arxiv paper](https://arxiv.org/abs/2310.08529v3) for details.
- 3/8/2024: We also provide a [GaussianDreamer extension for threestudio](https://github.com/cxh0519/threestudio-gaussiandreamer). Thanks for the contribution of [Xinhua Cheng](https://github.com/cxh0519/).
- 2/27/2024: Accepted by CVPR 2024.
- 12/6/2023: Update [arxiv paper](https://arxiv.org/abs/2310.08529).
- 11/27/2023: Update colab and huggingface demo.
- 11/27/2023: Release the results initialized using point clouds with ground. And now we support importing the generated 3D assets into the Unity game engine with the help of [UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting). See the [Project Page](https://taoranyi.com/gaussiandreamer/) for details.
- 10/24/2023: Release the results initialized using SMPL. See the [Project Page](https://taoranyi.com/gaussiandreamer/)  for details.
- 10/21/2023: Fixed some installation issues, thanks to Sikuang Li, [Tawfik Boujeh](), and [ashawkey](https://github.com/ashawkey/diff-gaussian-rasterization). You can view the detailed information in branch diff.
- 10/16/2023: The rough code has been released, and there may still be some issues. Please feel free to raise issues. 

## 😀 Demo
Huggingface demo: <a href="https://huggingface.co/spaces/thewhole/GaussianDreamer_Demo"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-Huggingface-orange"></a>

Colab demo: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/taoranyi/GaussianDreamer-colab/blob/main/GaussianDreamer_colab.ipynb) (Thanks [camenduru](https://github.com/camenduru/GaussianDreamer-colab).)


## 🚀 Get Started
**Installation**
Install [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [Shap-E](https://github.com/openai/shap-e#usage) as fellow:
```
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
pip install ninja
pip install -r requirements.txt

git clone https://github.com/hustvl/GaussianDreamer.git 
cd GaussianDreamer

pip install ./gaussiansplatting/submodules/diff-gaussian-rasterization
pip install ./gaussiansplatting/submodules/simple-knn

git clone https://github.com/openai/shap-e.git
cd shap-e
pip install -e .
```
Download [finetuned Shap-E](https://huggingface.co/datasets/tiange/Cap3D/blob/main/misc/our_finetuned_models/shapE_finetuned_with_330kdata.pth) by Cap3D, and put it in `./load`

**Quickstart**

Text-to-3D Generation
```
python launch.py --config configs/gaussiandreamer-sd.yaml --train --gpu 0 system.prompt_processor.prompt="a fox"

# if you want to import the generated 3D assets into the Unity game engine.
python launch.py --config configs/gaussiandreamer-sd.yaml --train --gpu 0 system.prompt_processor.prompt="a fox" system.sh_degree=3 
```

Text-to-Avatar Generation
```
python launch.py --config configs/gaussiandreamer-sd.yaml --train --gpu 0 system.prompt_processor.prompt="Spiderman stands with open arms" system.load_type=1

# if you want to import the generated 3D assets into the Unity game engine.
python launch.py --config configs/gaussiandreamer-sd.yaml --train --gpu 0 system.prompt_processor.prompt="Spiderman stands with open arms" system.load_type=1 system.sh_degree=3 
```


**Application**

Import the generated 3D assets into the Unity game engine to become materials for games and designs with the help of [UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting).
![block](./images/unity.gif)

## 🏁 Evaluation
We evaluate our model using the ViT similarity and [T<sup>3</sup>Bench](https://t3bench.com/), and the results are as follows.


###  ViT similarity
| **Methods** | **ViT-L/14 $\uparrow$** | **ViT-bigG-14 $\uparrow$** | **Generation Time $\downarrow$** |
| --- | --- | --- | --- |
| Shap-E | 20.51 | 32.21 | 6 seconds |
| DreamFusion| 23.60 | 37.46 | 1.5 hours |
| ProlificDreamer| 27.39 | 42.98 | 10 hours |
| Instant3D| 26.87 | 41.77 | 20 seconds |
| Ours | 27.23 $\pm$ 0.06 | 41.88 $\pm$ 0.04 | 15 minutes |
### [T<sup>3</sup>Bench](https://t3bench.com/)
| **Methods** | **Time** | **Single Obj.** | **Single w/ Surr.** | **Multi Obj.** | **Average** |
| --- | --- | --- | --- | --- | --- |
| SJC | -- | 24.7 | 19.8 | 11.7 | 18.7 |
| DreamFusion | 6 hours | 24.4 | 24.6 | 16.1 | 21.7 |
| Fantasia3D| 6 hours | 26.4 | 27.0 | 18.5 | 24.0 |
| LatentNeRF| 15 minutes | 33.1 | 30.6 | 20.6 | 28.1 |
| Magic3D| 5.3 hours | 37.0 | 35.4 | 25.7 | 32.7 |
| ProlificDreamer| 10 hours | 49.4 | 44.8 | **35.8** | 43.3 |
| Ours | 15 minutes | **54.0** | **48.6** | 34.5 | **45.7** |

## 📑 Citation
If you find this repository/work helpful in your research, welcome to cite the paper and give a ⭐.
Some source code of ours is borrowed from [Threestudio](https://github.com/threestudio-project/threestudio), [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [depth-diff-gaussian-rasterization](https://github.com/ingra14m/depth-diff-gaussian-rasterization). We sincerely appreciate the excellent works of these authors.
```
@inproceedings{yi2023gaussiandreamer,
  title={GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging 2D and 3D Diffusion Models},
  author={Yi, Taoran and Fang, Jiemin and Wang, Junjie and Wu, Guanjun and Xie, Lingxi and Zhang, Xiaopeng and Liu, Wenyu and Tian, Qi and Wang, Xinggang},
  year = {2024},
  booktitle = {CVPR}
}
```
