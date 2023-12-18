# Video Bypass Aggregator (VBA)

[中文](./README_zh-CN.md)

Undergraduate Thesis
[Download](./Understanding_Few-shot_Video_with_Pretrained_Image-Text_Models.pdf)

### Title: Understanding Few-shot Video with Pretrained Image-Text Models

### Advisor: [Jiwen Lu](https://scholar.google.com/citations?user=TN8uDQoAAAAJ&hl=en-US), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en), [Yansong Tang](https://andytang15.github.io/)

![VBA](resources/overall_intro.png)

### Abstract

Most existing few-shot video classification methods do not consider the transfer of pretrained knowledge in their design. The accuracy of these methods is sub-optimal when directly introducing pretrained knowledge, and the transfer efficiency is relatively poor. On the other hand, existing video classification methods for full-sample supervised learning always encode each sample independently, making it difficult to jointly encode all samples in the support and query sets in few-shot tasks.

To effectively transfer the scene modeling ability of pretrained models to the video domain and achieve fine accuracy in few-shot video classification tasks, this paper proposes the Video Bypass Aggregator (VBA) structure. Through cross-layer information aggregation, the method effectively transfers the multi-granularity scene modeling ability of pretrained features. Through cross-frame information aggregation, the method fully learns the temporal modeling ability for videos. Through cross-video information aggregation, the method uses all sample information in the support and query sets to obtain more robust feature representations for each sample.

The proposed method achieves higher or comparable accuracy compared to existing methods with the same settings on 6 mainstream video datasets and 7 few-shot splits, demonstrating the overall effectiveness of the method. The ablation experiments demonstrate the role of each module. The paper also demonstrates the training efficiency of the method, shows its cross-domain understanding ability, and provides qualitative visualizations of attention maps.

### Results

| Dataset   | Backbone        | Config          | Top-1 Acc <br>(Previous SOTA)  | Top-5 Acc <br>(Previous SOTA)  | Download CKPT |
|---------------|-----------|-------------------------------------------------------------------------------|---------------|------------|-----|
| Kinetics      | CLIP ViT-B/32  |  [config](/configs/fewshot/matchingnet/vit32_main/k_vit32_clip_freeze_5.py)   |  86.2 (82.0)  |   95.1 (91.4) | wget --no-check-certificate https://cloud.tsinghua.edu.cn/f/17c1e58c34144481967a/?dl=1 |
| minissv2_small| CLIP ViT-B/32  |  [config](/configs/fewshot/matchingnet/vit32_main/s_vit32_clip_temporal_5.py) |  52.1 (48.3)  |   71.2 (64.1) | wget --no-check-certificate https://cloud.tsinghua.edu.cn/f/2a560b10b1b54976afa6/?dl=1 |           
| minissv2_full | CLIP ViT-B/32  |  [config](/configs/fewshot/matchingnet/vit32_main/f_vit32_clip_temporal_5.py) |  61.5 (59.0)  |   74.7 (74.9) | wget --no-check-certificate https://cloud.tsinghua.edu.cn/f/25948c7b28a34ceab949/?dl=1 |          
| HMDB51        | CLIP ViT-B/32  |  [config](/configs/fewshot/matchingnet/vit32_main/h_vit32_clip_temporal_5.py) |  73.4 (71.1)  |   87.1 (83.9) | wget --no-check-certificate https://cloud.tsinghua.edu.cn/f/e76620555fb242b794a9/?dl=1 |          
| UCF101        | CLIP ViT-B/32  |  [config](/configs/fewshot/matchingnet/vit32_main/u_vit32_clip_freeze_5.py)   |  91.9 (92.6)  |   97.5 (96.8) | wget --no-check-certificate https://cloud.tsinghua.edu.cn/f/ae95b2f886934b288292/?dl=1 |           
| Diving48      | CLIP ViT-B/32  |  [config](/configs/fewshot/matchingnet/vit32_main/d_vit32_clip_temporal.py)   |  75.7 (74.4)  |   90.1 (85.6) | wget --no-check-certificate https://cloud.tsinghua.edu.cn/f/f6787f69207d4ace9c87/?dl=1 |           
| Gym99         | CLIP ViT-B/32  |  [config](/configs/fewshot/matchingnet/vit32_main/g_vit32_clip_temporal.py)   |  91.3 (90.0)  |   95.4 (93.6) | wget --no-check-certificate https://cloud.tsinghua.edu.cn/f/5bf3a6e177d840b4abdb/?dl=1 |           

* Previous SOTA here refers to [HyRSM](https://arxiv.org/abs/2204.13423) (CVPR 2022), with its backbone replaced by CLIP ViT-B/32.


### Prepare Environment

```bash
# prepare conda environment
conda create -n vba python=3.8
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
pip install decord scipy einops tensorboard imgaug setuptools==59.5.0 yapf==0.40.1 numpy==1.23.0
pip install -e .

# prepare pretrained weights
wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
mv ViT-B-32.pt root_dir/data/clip_weight/
```


### Prepare Data

This repo only provides few-shot data for Mini-Kinetics, Mini-SSV2, HMDB51, UCF101, and Diving48. For Gym99, please refer to its official repo.

```bash
# Mini-Kinetics, about 6G
cd root_dir/data/minikinetics
wget --no-check-certificate https://cloud.tsinghua.edu.cn/f/8b68f113b19a48338072/?dl=1
mv index.html\?dl\=1 minikinetics_videos.zip
unzip -q minikinetics_videos.zip

# Mini-SSV2, about 8G
cd root_dir/data/minissv2
wget --no-check-certificate https://cloud.tsinghua.edu.cn/f/bde18b48b9054143a1f4/?dl=1
mv index.html\?dl\=1 minissv2_videos.zip
unzip -q minissv2_videos.zip

# HMDB51, about 2G
cd root_dir/data/hmdb51
wget --no-check-certificate https://cloud.tsinghua.edu.cn/f/dd6fe5c35eb44b3fa2ca/?dl=1
mv index.html\?dl\=1 hmdb51_videos.zip
unzip -q hmdb51_videos.zip

# UCF101, about 7G
cd root_dir/data/ucf101
wget --no-check-certificate https://cloud.tsinghua.edu.cn/f/075698d33fcb4c6e84e9/?dl=1
mv index.html\?dl\=1 ucf101_videos.zip
unzip -q ucf101_videos.zip

# Diving48, about 5G
cd root_dir/data/diving48
wget --no-check-certificate https://cloud.tsinghua.edu.cn/f/d6e13e50c9b64bfca9b0/?dl=1
mv index.html\?dl\=1 diving48_videos.zip
unzip -q diving48_videos.zip
```

### Train and Test
```bash
python tools/train.py <config_path> --validate  # train
python tools/test.py <config_path> <ckpt_path>  # test
```
To get the exact results in the paper, please use the provided data & ckpt, and refer to `tools/train_and_test0.sh` and `tools/train_and_test1.sh`.
