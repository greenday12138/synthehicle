# Synthehicle

[![Paper](http://img.shields.io/badge/paper-arxiv.2208.14167-B31B1B.svg)](https://arxiv.org/abs/2208.14167)
[![Paper](https://img.shields.io/badge/paper-WACV2023-brightgreen)](https://openaccess.thecvf.com/content/WACV2023W/RWS/html/Herzog_Synthehicle_Multi-Vehicle_Multi-Camera_Tracking_in_Virtual_Cities_WACVW_2023_paper.html)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/synthehicle-multi-vehicle-multi-camera/multi-object-tracking-on-synthehicle)](https://paperswithcode.com/sota/multi-object-tracking-on-synthehicle)

![gif](synthehicle.gif)

Synthehicle is a massive CARLA-based synthehic multi-vehicle multi-camera tracking dataset and includes ground truth for 2D detection and tracking, 3D detection and tracking, depth estimation, and semantic, instance and panoptic segmentation. 

### News (18/11/22)
 
* Synthehicle has been accepted to WACV Workshops 2023
* We have added the CARLA and evaluation scripts
* The evaluation server is ready!

## Dataset

The 17 hour Synthehicle dataset consists of 64 scenes in four different weather conditions, 16 different camera setups, and 340 camera videos. It is freely available via the following download links provided [here](DOWNLOAD.md).

## Evaluation

To evaluate on Synthehicle please refer to our [wiki](https://github.com/fubel/synthehicle/wiki/Evaluation).

## Generate Data

To generate your own data similar to Synthehicle please refer to our [wiki](https://github.com/fubel/synthehicle/wiki/Data-Creation).

## Pretrained Models

We provide pretrained weights for 2D detection and vehicle re-identification:

### Detection 
We have used the YOLOX-x model from mmdetection.

| Model   | Trained on | Weights                                                                                                                                                                     | Config                                                                                                                                                                   | AP    |
| ------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----- |
| YOLOX-x | All        | [download](https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=18e2eac4&path=Datenbanken\Synthehicle\pretrained_models\detection\yolox_synthehicle_all.pth)   | [download](https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=18e2eac4&path=Datenbanken\Synthehicle\pretrained_models\detection\yolox_x_8x8_300e_coco.py) | 59.7% |
| YOLOX-x | Day        | [download](https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=18e2eac4&path=Datenbanken\Synthehicle\pretrained_models\detection\yolox_synthehicle_day.pth)   | [download](https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=18e2eac4&path=Datenbanken\Synthehicle\pretrained_models\detection\yolox_x_8x8_300e_coco.py) | 58.7% |
| YOLOX-x | Dawn       | [download](https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=18e2eac4&path=Datenbanken\Synthehicle\pretrained_models\detection\yolox_synthehicle_dawn.pth)  | [download](https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=18e2eac4&path=Datenbanken\Synthehicle\pretrained_models\detection\yolox_x_8x8_300e_coco.py) | 60.8% |
| YOLOX-x | Rain       | [download](https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=18e2eac4&path=Datenbanken\Synthehicle\pretrained_models\detection\yolox_synthehicle_rain.pth)  | [download](https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=18e2eac4&path=Datenbanken\Synthehicle\pretrained_models\detection\yolox_x_8x8_300e_coco.py) | 56.8% |
| YOLOX-x | Night      | [download](https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=18e2eac4&path=Datenbanken\Synthehicle\pretrained_models\detection\yolox_synthehicle_night.pth) | [download](https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=18e2eac4&path=Datenbanken\Synthehicle\pretrained_models\detection\yolox_x_8x8_300e_coco.py) | 50.6% |

The specialized models (day, dawn, rain, night) are provided for completeness. Results from our paper indicate that the model trained on all subsets performs best for all environmental setups.

### Vehicle Re-Identification

We have used the fastreid ResNet-50 Model with IBN:

| Model    | Trained on | Weights                                                                                                                                                          | Config | mAP    |
| -------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ------ |
| fastreid | All        | [download](https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=18e2eac4&path=Datenbanken\Synthehicle\pretrained_models\reid\synthehicle_all.pth)   |        | 47.8%  |
| fastreid | Day        | [download](https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=18e2eac4&path=Datenbanken\Synthehicle\pretrained_models\reid\synthehicle_day.pth)   |        | 59.8%  |
| fastreid | Dawn       | [download](https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=18e2eac4&path=Datenbanken\Synthehicle\pretrained_models\reid\synthehicle_dawn.pth)  |        | 47.57% |
| fastreid | Rain       | [download](https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=18e2eac4&path=Datenbanken\Synthehicle\pretrained_models\reid\synthehicle_rain.pth)  |        | 39.08% |
| fastreid | Night      | [download](https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=18e2eac4&path=Datenbanken\Synthehicle\pretrained_models\reid\synthehicle_night.pth) |        | 27.04% |

The specialized models (day, dawn, rain, night) are provided for completeness. Results from our paper indicate that the model trained on all subsets performs best for all environmental setups. We will provide a fast-reid config soon alongside a model class. The weights can be read into any fast-reid ResNet-50 model.

## Tracking

In our paper, single-camera tracking has been performed using [DeepSORT](https://github.com/nwojke/deep_sort) with the models above. Multi-camera tracking has been performed using [ELECTRICITY](https://github.com/KevinQian97/ELECTRICITY-MTMC).

## Citation   

If you use Synthehicle for your work, please cite:

```
@InProceedings{Herzog_2023_WACV,
    author    = {Herzog, Fabian and Chen, Junpeng and Teepe, Torben and Gilg, Johannes and H\"ormann, Stefan and Rigoll, Gerhard},
    title     = {Synthehicle: Multi-Vehicle Multi-Camera Tracking in Virtual Cities},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
    month     = {January},
    year      = {2023},
    pages     = {1-11}
}
``` 
