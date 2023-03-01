# AP-PA (Adaptive-Patch-based Physical Attack)

## Introduction

In this paper, a novel adaptive-patch-based physical attack (AP-PA) framework is proposed, which aims to generate adversarial patches that are adaptive in both physical dynamics and varying scales, and by which the particular targets can be hidden from being detected. Furthermore, the adversarial patch is also gifted with attack effectiveness against all targets of the same class with a patch outside the target (No need to smear targeted objects) and robust enough in the physical world. In addition, a new loss is devised to consider more available information of detected objects to optimize the adversarial patch, which can significantly improve the patch's attack efficacy (Average precision drop up to $87.86\%$ and $85.48\%$ in white-box and black-box settings, respectively) and optimizing efficiency. We also establish one of the first comprehensive, coherent, and rigorous benchmarks to evaluate the attack efficacy of adversarial patches on aerial detection tasks. We summarize our algorithm in [Benchmarking Adversarial Patch Against Aerial Detection](https://ieeexplore.ieee.org/document/9965436).

## Requirements:

* Pytorch 1.10

* Python 3.6

## Citation

If you use AP-PA method for attacks in your research, please consider citing

```
@article{lian2022benchmarking,
  title={Benchmarking Adversarial Patch Against Aerial Detection},
  author={Lian, Jiawei and Mei, Shaohui and Zhang, Shun and Ma, Mingyang},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--16},
  year={2022},
  publisher={IEEE}
}
```
