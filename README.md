This repository is folked from [https://github.com/yjh0410/DetLAB](https://github.com/yjh0410/DetLAB).
At this repository, simplification and explanation and will be tested at Colab Environment. Some bugs were fixed.

# 1. Retinanet

## Engilish
*  **Theory** : [https://wikidocs.net/227235](https://wikidocs.net/227235) <br>
*  **Implementation** : [https://wikidocs.net/227237](https://wikidocs.net/227237)

## 한글
*  **Theory** : [https://wikidocs.net/225902](https://wikidocs.net/225902) <br>
*  **Implementation** : [https://wikidocs.net/226020](https://wikidocs.net/226020)


## Main results on COCO-val
#### RetinaNet
| Model                          |  scale     |   mAP   | Weight|
|--------------------------------|------------|---------|-------|
| RetinaNet_R_18_1x              |  800,1333  |   29.3  | [github](https://github.com/yjh0410/DetLAB/releases/download/object-detection-benchmark-weight/retinanet_r18_1x_29.3.pth) |
| RetinaNet_R_50_1x              |  800,1333  |   35.8  | [github](https://github.com/yjh0410/DetLAB/releases/download/object-detection-benchmark-weight/retinanet_r50_1x_35.8.pth) |
| RetinaNet-RT_R_50_3x           |  512,736   |   32.0  | [github](https://github.com/yjh0410/DetLAB/releases/download/object-detection-benchmark-weight/retinanet-rt_r50_3x_32.0.pth) |

In RetinaNet:
- For regression head, `GIoU Loss` is deployed rather than `SmoothL1Loss`


## Step 1. Clone from Github and install library

Git clone to root directory. 

```Shell
# Clone from Github Repository
! git init .
! git remote add origin https://github.com/RichardMinsooGo-ML/Bible_4_Part_J_Retina_FCOS_YOLOF.git
# ! git pull origin master
! git pull origin main
```

A tool to count the FLOPs of PyTorch model.

```
from IPython.display import clear_output
clear_output()
```

```Shell
! pip install thop
```

## Step x. Download pretrained weight

```Shell
! wget https://github.com/yjh0410/DetLAB/releases/download/object-detection-benchmark-weight/retinanet_r18_1x_29.3.pth
! wget https://github.com/yjh0410/DetLAB/releases/download/object-detection-benchmark-weight/retinanet_r50_1x_35.8.pth
! wget https://github.com/yjh0410/DetLAB/releases/download/object-detection-benchmark-weight/retinanet-rt_r50_3x_32.0.pth
```

## Demo


#### detect images
I have provide some images in `/content/dataset/demo/images/`, so you can run following command to run a demo:

```Shell
# Detect with Image

! python demo_Retina.py --mode image \
                 --path_to_img /content/dataset/demo/images/ \
                 -v retinanet50 \
                 --cuda \
                 --weight /content/retinanet_r50_1x_35.8.pth
                 # --show

# Check result at : /content/det_results/images/image
```

#### detect video
If you want run a demo of streaming video detection, you need to set `--mode` to `video`, and give the path to video `--path_to_vid`。

```Shell
# Detect with Video

! python demo_Retina.py --mode video \
                 --path_to_vid /content/dataset/demo/videos/street.mp4 \
                 -v retinanet50 \
                 --cuda \
                 --weight /content/retinanet_r50_1x_35.8.pth
                 # --show

# Check result at : /content/det_results/images/video
```

#### detect at camera
If you want run video detection with your camera, you need to set `--mode` to `camera`。

```Shell
# Detect with Camera
# it don't work at Colab. Use laptop

# ! python demo_Retina.py --mode video \
#                  --path_to_img data/demo/videos/your_video \
#                  -v retinanet50 \
#                  --cuda \
#                  --weight /content/retinanet_r50_1x_35.8.pth
                 # --show
```


## Download COCO Dataset

```Shell
# COCO dataset download and extract

# ! wget http://images.cocodataset.org/zips/train2017.zip
! wget http://images.cocodataset.org/zips/val2017.zip
! wget http://images.cocodataset.org/zips/test2017.zip
# ! wget http://images.cocodataset.org/zips/unlabeled2017.zip

# ! unzip train2017.zip  -d dataset/COCO
! unzip val2017.zip  -d dataset/COCO
! unzip test2017.zip  -d dataset/COCO

# ! unzip unlabeled2017.zip -d dataset/COCO

# ! rm train2017.zip
# ! rm val2017.zip
# ! rm test2017.zip
# ! rm unlabeled2017.zip

! wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
# wget http://images.cocodataset.org/annotations/image_info_test2017.zip
# wget http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip

! unzip annotations_trainval2017.zip -d dataset/COCO
# ! unzip stuff_annotations_trainval2017.zip
# ! unzip image_info_test2017.zip
# ! unzip image_info_unlabeled2017.zip

# ! rm annotations_trainval2017.zip
# ! rm stuff_annotations_trainval2017.zip
# ! rm image_info_test2017.zip
# ! rm image_info_unlabeled2017.zip

clear_output()
```

## Test 

```Shell
! python test.py -d coco \
                 --cuda \
                 -v retinanet50 \
                 --weight /content/retinanet_r50_1x_35.8.pth \
                 --root /content/dataset
                 # --show
```

## Download VOC Dataset

```Shell
# VOC 2012 Dataset Download and extract

! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
!tar -xvf "/content/VOCtrainval_11-May-2012.tar" -C "/content/dataset"
clear_output()

# VOC 2007 Dataset Download and extract

! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
!tar -xvf "/content/VOCtrainval_06-Nov-2007.tar" -C "/content/dataset"
!tar -xvf "/content/VOCtest_06-Nov-2007.tar" -C "/content/dataset"
clear_output()
```

## Train at single GPU-voc dataset

```Shell
# batch_sze 32, T4 GPU memory 5.7GB

! python train.py \
        --cuda \
        -d voc \
        --root /content/dataset \
        -v retinanet18 \
        --lr_scheduler step \
        --schedule 1x \
        --grad_clip_norm 4.0
```

```Shell
# batch_size 16, A100-GPU memory 30.4GB

! python train.py \
        --cuda \
        -d voc \
        --root /content/dataset \
        -v retinanet50 \
        --lr_scheduler step \
        --schedule 1x \
        --grad_clip_norm 4.0
```

```Shell
# batch_size 16, A-100 GPU memory 38.1 GB

! python train.py \
        --cuda \
        -d voc \
        --root /content/dataset \
        -v retinanet101 \
        --lr_scheduler step \
        --schedule 1x \
        --grad_clip_norm 4.0
```

```Shell
# Not Working at Colab Pro +

# ! python train.py \
#         --cuda \
#         -d voc \
#         --root /content/dataset \
#         -v retinanet-rt \
#         --lr_scheduler step \
#         --schedule 1x \
#         --batch_size 16 \
#         --grad_clip_norm 4.0
```

## Train at Multi GPU-voc dataset

```Shell
# Not Working at Colab Pro +
# 2 GPUs
# Attention, the following batch size is on single GPU, not all GPUs.
# ! python -m torch.distributed.run --nproc_per_node=2 train.py \
#                                                     --cuda \
#                                                     -dist \
#                                                     -d coco \
#                                                     --root /mnt/share/ssd2/dataset/ \
#                                                     -v fcos50 \
#                                                     -lr 0.01 \
#                                                     -lr_bk 0.01 \
#                                                     --batch_size 8 \
#                                                     --grad_clip_norm 4.0 \
#                                                     --num_workers 4 \
#                                                     --schedule 1x
                                                    # --sybn
                                                    # --mosaic
```

# 2. FCOS

## Engilish
*  **Theory** : [https://wikidocs.net/227238](https://wikidocs.net/227238) <br>
*  **Implementation** : [https://wikidocs.net/227239](https://wikidocs.net/227239)

## 한글
*  **Theory** : [https://wikidocs.net/226018](https://wikidocs.net/226018) <br>
*  **Implementation** : [https://wikidocs.net/225907](https://wikidocs.net/225907)

## Main results on COCO-val
#### FCOS
| Model                          |  scale     |   mAP   | Weight|
|--------------------------------|------------|---------|-------|
| FCOS_R_18_1x                   |  800,1333  |  31.3   | [github](https://github.com/yjh0410/DetLAB/releases/download/object-detection-benchmark-weight/fcos_r18_1x_31.3.pth) |
| FCOS_R_50_1x                   |  800,1333  |  37.6   | [github](https://github.com/yjh0410/DetLAB/releases/download/object-detection-benchmark-weight/fcos_r50_1x_37.6.pth) |
| FCOS-RT_R_50_OTA_3x            |  640,640   |  36.7   | [github](https://github.com/yjh0410/DetLAB/releases/download/object-detection-benchmark-weight/fcos-rt-ota_r50_3x_36.7.pth) |

In FCOS:
- For regression head, `GIoU loss` is deployed rather than `IoU loss`
- For real-time FCOS, the `PaFPN` is deployed for fpn


## Step 1. Clone from Github and install library

Git clone to root directory. 

```Shell
# Clone from Github Repository
! git init .
! git remote add origin https://github.com/RichardMinsooGo-ML/Bible_4_Part_J_Retina_FCOS_YOLOF.git
# ! git pull origin master
! git pull origin main
```

A tool to count the FLOPs of PyTorch model.

```
from IPython.display import clear_output
clear_output()
```

```Shell
! pip install thop
```

## Step x. Download pretrained weight

```Shell
! wget https://github.com/yjh0410/DetLAB/releases/download/object-detection-benchmark-weight/fcos_r18_1x_31.3.pth
! wget https://github.com/yjh0410/DetLAB/releases/download/object-detection-benchmark-weight/fcos_r50_1x_37.6.pth
! wget https://github.com/yjh0410/DetLAB/releases/download/object-detection-benchmark-weight/fcos-rt-ota_r50_3x_36.7.pth
```

## Demo

#### detect images
I have provide some images in `/content/dataset/demo/images/`, so you can run following command to run a demo:

```Shell
# Detect with Image

! python demo_FCOS.py --mode image \
                 --path_to_img /content/dataset/demo/images/ \
                 -v fcos50 \
                 --cuda \
                 --weight /content/fcos_r50_1x_37.6.pth
                 # --show

# Check result at : /content/det_results/images/image
```

#### detect video
If you want run a demo of streaming video detection, you need to set `--mode` to `video`, and give the path to video `--path_to_vid`。

```Shell
# Detect with Video

! python demo_FCOS.py --mode video \
                 --path_to_vid /content/dataset/demo/videos/street.mp4 \
                 -v fcos50 \
                 --cuda \
                 --weight /content/fcos_r50_1x_37.6.pth
                 # --show

# Check result at : /content/det_results/images/video
```

#### detect at camera
If you want run video detection with your camera, you need to set `--mode` to `camera`。

```Shell
# Detect with Camera
# it don't work at Colab. Use laptop

# ! python demo_FCOS.py --mode video \
#                  --path_to_img data/demo/videos/your_video \
#                  -v fcos50 \
#                  --cuda \
#                  --weight /content/fcos_r50_1x_37.6.pth
                 # --show
```


## Download COCO Dataset

```Shell
# COCO dataset download and extract

# ! wget http://images.cocodataset.org/zips/train2017.zip
! wget http://images.cocodataset.org/zips/val2017.zip
! wget http://images.cocodataset.org/zips/test2017.zip
# ! wget http://images.cocodataset.org/zips/unlabeled2017.zip

# ! unzip train2017.zip  -d dataset/COCO
! unzip val2017.zip  -d dataset/COCO
! unzip test2017.zip  -d dataset/COCO

# ! unzip unlabeled2017.zip -d dataset/COCO

# ! rm train2017.zip
# ! rm val2017.zip
# ! rm test2017.zip
# ! rm unlabeled2017.zip

! wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
# wget http://images.cocodataset.org/annotations/image_info_test2017.zip
# wget http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip

! unzip annotations_trainval2017.zip -d dataset/COCO
# ! unzip stuff_annotations_trainval2017.zip
# ! unzip image_info_test2017.zip
# ! unzip image_info_unlabeled2017.zip

# ! rm annotations_trainval2017.zip
# ! rm stuff_annotations_trainval2017.zip
# ! rm image_info_test2017.zip
# ! rm image_info_unlabeled2017.zip

clear_output()
```

## Test 

```Shell
! python test.py -d coco \
                 --cuda \
                 -v fcos50 \
                 --weight /content/fcos_r50_1x_37.6.pth \
                 --root /content/dataset
                 # --show
```

## Download VOC Dataset

```Shell
# VOC 2012 Dataset Download and extract

! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
!tar -xvf "/content/VOCtrainval_11-May-2012.tar" -C "/content/dataset"
clear_output()

# VOC 2007 Dataset Download and extract

! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
!tar -xvf "/content/VOCtrainval_06-Nov-2007.tar" -C "/content/dataset"
!tar -xvf "/content/VOCtest_06-Nov-2007.tar" -C "/content/dataset"
clear_output()
```

## Train at single GPU-voc dataset

```Shell
# T-4GPU RAM max max 13.6GB, avg. 12.4 GB, for 5 epochs
! python train.py \
        --cuda \
        -d voc \
        --root /content/dataset \
        -v fcos18 \
        --lr_scheduler step \
        --schedule 1x \
        --batch_size 16 \
        --grad_clip_norm 4.0
```

```Shell
# Use A-100 GPU batch_size = 8 / GPU RAM max 27.6GB
! python train.py \
        --cuda \
        -d voc \
        --root /content/dataset \
        -v fcos50 \
        --lr_scheduler step \
        --schedule 1x \
        --batch_size 8 \
        --grad_clip_norm 4.0
```

```Shell
# Not Working at Colab
# ! python train.py \
#         --cuda \
#         -d voc \
#         --root /content/dataset \
#         -v fcos50-ota \
#         --lr_scheduler step \
#         --schedule 1x \
#         --batch_size 16 \
#         --grad_clip_norm 4.0
```

```Shell
# Use A-100 GPU batch_size = 8 / GPU RAM max 38.8G
! python train.py \
        --cuda \
        -d voc \
        --root /content/dataset \
        -v fcos101 \
        --lr_scheduler step \
        --schedule 1x \
        --batch_size 8 \
        --grad_clip_norm 4.0
```

```Shell
# Not Working at Colab
# ! python train.py \
#         --cuda \
#         -d voc \
#         --root /content/dataset \
#         -v fcos101-ota \
#         --lr_scheduler step \
#         --schedule 1x \
#         --batch_size 16 \
#         --grad_clip_norm 4.0
```


```Shell
# Not Working at Colab Pro +
# ! python train.py \
#         --cuda \
#         -d voc \
#         --root /content/dataset \
#         -v fcos-rt-ota \
#         --lr_scheduler step \
#         --schedule 1x \
#         --batch_size 16 \
#         --grad_clip_norm 4.0
```

## Train at Multi GPU-voc dataset

```Shell
# Not Working at Colab Pro +
# 2 GPUs
# Attention, the following batch size is on single GPU, not all GPUs.
# ! python -m torch.distributed.run --nproc_per_node=2 train.py \
#                                                     --cuda \
#                                                     -dist \
#                                                     -d coco \
#                                                     --root /mnt/share/ssd2/dataset/ \
#                                                     -v fcos50 \
#                                                     -lr 0.01 \
#                                                     -lr_bk 0.01 \
#                                                     --batch_size 8 \
#                                                     --grad_clip_norm 4.0 \
#                                                     --num_workers 4 \
#                                                     --schedule 1x
                                                    # --sybn
                                                    # --mosaic
```



# 3. YoloF

## Engilish
*  **Theory** : [https://wikidocs.net/227240](https://wikidocs.net/227240) <br>
*  **Implementation** : [https://wikidocs.net/227241](https://wikidocs.net/227241)

## 한글
*  **Theory** : [https://wikidocs.net/226019](https://wikidocs.net/226019) <br>
*  **Implementation** : [https://wikidocs.net/225908](https://wikidocs.net/225908)

## Main results on COCO-val
#### YOLOF

| Model                          |  scale     |   mAP   | Weight|
|--------------------------------|------------|---------|-------|
| YOLOF_R_18_C5_1x               |  800,1333  |   31.6  | [github](https://github.com/yjh0410/DetLAB/releases/download/object-detection-benchmark-weight/yolof_r18_C5_1x_31.6.pth) |
| YOLOF_R_50_C5_1x               |  800,1333  |   37.6  | [github](https://github.com/yjh0410/DetLAB/releases/download/object-detection-benchmark-weight/yolof_r50_C5_1x_37.6.pth) |
| YOLOF-RT_R_50_DC5_3x           |  640,640   |   38.1  | [github](https://github.com/yjh0410/DetLAB/releases/download/object-detection-benchmark-weight/yolof-rt_r50_DC5_1x_38.1.pth) |


## Step 1. Clone from Github and install library

Git clone to root directory. 

```Shell
# Clone from Github Repository
! git init .
! git remote add origin https://github.com/RichardMinsooGo-ML/Bible_4_Part_J_Retina_FCOS_YOLOF.git
# ! git pull origin master
! git pull origin main
```

A tool to count the FLOPs of PyTorch model.

```
from IPython.display import clear_output
clear_output()
```

```Shell
! pip install thop
```

## Step x. Download pretrained weight

```Shell
! wget https://github.com/yjh0410/DetLAB/releases/download/object-detection-benchmark-weight/yolof_r18_C5_1x_31.6.pth
! wget https://github.com/yjh0410/DetLAB/releases/download/object-detection-benchmark-weight/yolof_r50_C5_1x_37.6.pth
! wget https://github.com/yjh0410/DetLAB/releases/download/object-detection-benchmark-weight/yolof-rt_r50_DC5_1x_38.1.pth
```

## Demo


#### detect images
I have provide some images in `/content/dataset/demo/images/`, so you can run following command to run a demo:

```Shell
# Detect with Image

! python demo_YoloF.py --mode image \
                 --path_to_img /content/dataset/demo/images/ \
                 -v yolof50 \
                 --cuda \
                 --weight /content/yolof_r50_C5_1x_37.6.pth
                 # --show

# Check result at : /content/det_results/images/image
```

#### detect video
If you want run a demo of streaming video detection, you need to set `--mode` to `video`, and give the path to video `--path_to_vid`。

```Shell
# Detect with Video

! python demo_YoloF.py --mode video \
                 --path_to_vid /content/dataset/demo/videos/street.mp4 \
                 -v yolof50 \
                 --cuda \
                 --weight /content/yolof_r50_C5_1x_37.6.pth
                 # --show

# Check result at : /content/det_results/images/video
```

#### detect at camera
If you want run video detection with your camera, you need to set `--mode` to `camera`。

```Shell
# Detect with Camera
# it don't work at Colab. Use laptop

# ! python demo_YoloF.py --mode video \
#                  --path_to_img data/demo/videos/your_video \
#                  -v yolof50 \
#                  --cuda \
#                  --weight /content/yolof_r50_C5_1x_37.6.pth
                 # --show
```


## Download COCO Dataset

```Shell
# COCO dataset download and extract

# ! wget http://images.cocodataset.org/zips/train2017.zip
! wget http://images.cocodataset.org/zips/val2017.zip
! wget http://images.cocodataset.org/zips/test2017.zip
# ! wget http://images.cocodataset.org/zips/unlabeled2017.zip

# ! unzip train2017.zip  -d dataset/COCO
! unzip val2017.zip  -d dataset/COCO
! unzip test2017.zip  -d dataset/COCO

# ! unzip unlabeled2017.zip -d dataset/COCO

# ! rm train2017.zip
# ! rm val2017.zip
# ! rm test2017.zip
# ! rm unlabeled2017.zip

! wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
# wget http://images.cocodataset.org/annotations/image_info_test2017.zip
# wget http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip

! unzip annotations_trainval2017.zip -d dataset/COCO
# ! unzip stuff_annotations_trainval2017.zip
# ! unzip image_info_test2017.zip
# ! unzip image_info_unlabeled2017.zip

# ! rm annotations_trainval2017.zip
# ! rm stuff_annotations_trainval2017.zip
# ! rm image_info_test2017.zip
# ! rm image_info_unlabeled2017.zip

clear_output()
```

## Test 

```Shell
! python test.py -d coco \
                 --cuda \
                 -v yolof50 \
                 --weight /content/yolof_r50_C5_1x_37.6.pth \
                 --root /content/dataset
                 # --show
```

## Download VOC Dataset

```Shell
# VOC 2012 Dataset Download and extract

! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
!tar -xvf "/content/VOCtrainval_11-May-2012.tar" -C "/content/dataset"
clear_output()

# VOC 2007 Dataset Download and extract

! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
!tar -xvf "/content/VOCtrainval_06-Nov-2007.tar" -C "/content/dataset"
!tar -xvf "/content/VOCtest_06-Nov-2007.tar" -C "/content/dataset"
clear_output()
```

## Train at single GPU-voc dataset

```Shell
# T4-GPU 8.2GB
! python train.py \
        --cuda \
        -d voc \
        --root /content/dataset \
        -v yolof18 \
        --lr_scheduler step \
        --schedule 1x \
        --batch_size 16 \
        --grad_clip_norm 4.0
```

```Shell
# Use A-100 GPU RAM 20.2 GB
! python train.py \
        --cuda \
        -d voc \
        --root /content/dataset \
        -v yolof50 \
        --lr_scheduler step \
        --schedule 1x \
        --batch_size 16 \
        --grad_clip_norm 4.0
```

```Shell
# Use A-100 GPU batch_size = 8 / GPU RAM max ? G / avg. 28.8G
! python train.py \
        --cuda \
        -d voc \
        --root /content/dataset \
        -v yolof50-DC5 \
        --lr_scheduler step \
        --schedule 1x \
        --batch_size 8 \
        --grad_clip_norm 4.0
```

```Shell
# Use A-100 GPU RAM 31.5 GB
! python train.py \
        --cuda \
        -d voc \
        --root /content/dataset \
        -v yolof101 \
        --lr_scheduler step \
        --schedule 1x \
        --batch_size 8 \
        --grad_clip_norm 4.0
```

```Shell
# Use A-100 GPU RAM 37.8 GB
! python train.py \
        --cuda \
        -d voc \
        --root /content/dataset \
        -v yolof101-DC5 \
        --lr_scheduler step \
        --schedule 1x \
        --batch_size 8 \
        --grad_clip_norm 4.0
```

```Shell
# GPU RAM 8.5GB
! python train.py \
        --cuda \
        -d voc \
        --root /content/dataset \
        -v yolof-rt \
        --lr_scheduler step \
        --schedule 1x \
        --batch_size 16 \
        --grad_clip_norm 4.0
```

```Shell
# Not Working at Colab
# ! python train.py \
#         --cuda \
#         -d voc \
#         --root /content/dataset \
#         -v yolof18_exp \
#         --lr_scheduler step \
#         --schedule 1x \
#         --batch_size 16 \
#         --grad_clip_norm 4.0
```

## Train at Multi GPU-voc dataset

```Shell
# Not Working at Colab Pro +
# 2 GPUs
# Attention, the following batch size is on single GPU, not all GPUs.
# ! python -m torch.distributed.run --nproc_per_node=2 train.py \
#                                                     --cuda \
#                                                     -dist \
#                                                     -d coco \
#                                                     --root /mnt/share/ssd2/dataset/ \
#                                                     -v fcos50 \
#                                                     -lr 0.01 \
#                                                     -lr_bk 0.01 \
#                                                     --batch_size 8 \
#                                                     --grad_clip_norm 4.0 \
#                                                     --num_workers 4 \
#                                                     --schedule 1x
                                                    # --sybn
                                                    # --mosaic
```
