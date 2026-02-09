# Volume-C-CAM
This is the official PyTorch implementation of our paper "Divide and Conquer: Volume-wise Causal CAM for Weakly Supervised Semantic Segmentation on Medical Image"

## Requirement
- Pytorch 1.7.1, torchvision 0.8.2, and more in requirements.txt
- Dataset: [ProMRI](https://pan.baidu.com/s/1Sddlz9tP5C-dn6D9ITeuQw?pwd=7cvo), [ACDC](https://acdc.creatis.insa-lyon.fr/#challenge/5846c3366a3c7735e84b67ec) and [CHAOS](https://zenodo.org/record/3431873#.Ys7GRtpBy4R)
- 2 NVIDIA GPUs, and each has more than 11GB of memory

## Usage
1.Train the pure cam model, generate CAM, and evaluate the CAM.

```bash
cd myExperiment/
python run_pure_cam.py
```
note: directly run the eval_cam.py can compute the score with the best threshold.

2.(Optional) Refine the pure CAM, generate coarse segmentation mask.
```bash
python run_pure_refine.py
```
note: if do not run this step, directly generate coarse mask by running the eval_cam.py (change the file path by yourself).

3.compute confounders.
```bash
python run_get_confounder.py
```

4.train the causal cam model.
```bash
python run_causal_cam.py
```

5.(Optional) filter unreasonable positions in CAM with anatomy prior.
```bash
cd tools/
python cam_filter.py
```

6.(Optional) Refine causal CAM, generate pseudo mask.
```bash
python run_causal_refine.py
```

7.(Optional) Refine causal CAM with volume-wise strategy, generate pseudo mask.
```bash
python run_causal_refine_volume.py
```
note: if do not run step 6 and 7, directly generate pseudo mask by running the eval_cam.py (change the file path by yourself).


8.Use the generated pseudo mask (in round1/prediction/sem_seg_label/) to train a U-Net.
