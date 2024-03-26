

Thanks to the work of [YudeWang](https://github.com/YudeWang), the code of this repository borrow heavly from his [SEAM](https://github.com/YudeWang/SEAM) repository, and we follw the same pipeline to verify the effectiveness of our DAEN.

## Requirements
- Python 3.6
- pytorch 1.2.0, torchvision 0.2.1
- CUDA 10.1
- 3 x GPUs (12GB)

## Usage
### Installation
- Download the repository.
```
git clone https://github.com/YudeWang/SEAM.git
```
- Install python dependencies.
```
pip install -r requirements.txt
```
- **Download model weights from [google drive](https://drive.google.com/open?id=1jWsV5Yev-PwKgvvtUM3GnY0ogb50-qKa) or [baidu cloud](https://pan.baidu.com/s/1ymaMeF0ASjQ9oCGI9cmqHQ) (with code 6nmo)**, including ImageNet pretrained models and our training results.

- Download PASCAL VOC 2012 devkit (follow instructions in http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit). It is suggested to make a soft link toward downloaded dataset.
```
ln -s $your_dataset_path/VOCdevkit/VOC2012 VOC2012
```

- (Optional) The image-level labels have already been given in `voc12/cls_label.npy`. If you want to regenerate it (which is unnecessary), please download the annotation of VOC 2012 SegmentationClassAug training set (containing 10582 images), which can be download [here](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) and place them all as `VOC2012/SegmentationClassAug/xxxxxx.png`. Then run the code
```
cd voc12
python make_cls_labels.py --voc12_root VOC2012
```
### DAEN step

1. DAEN training
```
python train_DAEN.py --voc12_root VOC2012 --weights $pretrained_model --session_name $your_session_name
```

2. DAEN inference. 
```
python infer_DAEN.py --weights $SEAM_weights --infer_list [voc12/val.txt | voc12/train.txt | voc12/train_aug.txt] --out_cam $your_cam_dir --out_crf $your_crf_dir
```

3. DAEN step evaluation. We provide python mIoU evaluation script `evaluation.py`, or you can use official development kit. Here we suggest to show the curve of mIoU with different background score.
```
python evaluation.py --list VOC2012/ImageSets/Segmentation/[val.txt | train.txt] --predict_dir $your_cam_dir --gt_dir VOC2012/SegmentationClass --comment $your_comments --type npy --curve True
```

### Random walk step
The random walk step keep the same with AffinityNet repository.
1. Train AffinityNet.
```
python train_aff.py --weights $pretrained_model --voc12_root VOC2012 --la_crf_dir $your_crf_dir_4.0 --ha_crf_dir $your_crf_dir_24.0 --session_name $your_session_name
```
2. Random walk propagation
```
python infer_aff.py --weights $aff_weights --infer_list [voc12/val.txt | voc12/train.txt] --cam_dir $your_cam_dir --voc12_root VOC2012 --out_rw $your_rw_dir
```
3. Random walk step evaluation
```
python evaluation.py --list VOC2012/ImageSets/Segmentation/[val.txt | train.txt] --predict_dir $your_rw_dir --gt_dir VOC2012/SegmentationClass --comment $your_comments --type png
```

### Pseudo labels retrain
Pseudo label retrain on DeepLabv1. Code is available [here](https://github.com/YudeWang/semantic-segmentation-codebase/tree/main/experiment/seamv1-pseudovoc).


