## Example: Distributed MobileNet

### Device Setup
- TF 1.3 installed
- Subset of ImageNet dataset on device:
```shell
/home/pi/imagenet-data $ ls 
labels.txt                 validation-00003-of-00128  validation-00007-of-00128
validation-00000-of-00128  validation-00004-of-00128  validation-00008-of-00128
validation-00001-of-00128  validation-00005-of-00128  validation-00009-of-00128
validation-00002-of-00128  validation-00006-of-00128
```
- [Mobilenet v1 (0.25)](http://download.tensorflow.org/models/mobilenet_v1_0.25_128_2017_06_14.tar.gz) checkpoint on device:
```shell
/home/pi/mobilenet-0.25 $ ls
mobilenet_v1_0.25_128.ckpt.data-00000-of-00001
mobilenet_v1_0.25_128.ckpt.index
mobilenet_v1_0.25_128.ckpt.meta
```
- Run [`launch_device.py`](bin/launch_device.py):
```shell
/home/pi $ python launch_device.py --server SERVER --device DEVICE
```

### Server Setup
- TF 1.3 installed
- Mobilenet v1 (0.25) checkpoint on server in folder `(research/slim/)mobilenet-0.25`
- Main run command:

```shell
research/slim $ python eval_image_classifier.py \
  --alsologtostderr \
  --checkpoint_path=mobilenet-0.25/mobilenet_v1_0.25_128.ckpt \
  --dataset_name=imagenet \
  --dataset_split_name=validation \
  --model_name=mobilenet_v1_025 \
  --preprocessing_name=mobilenet_v1 \
  --batch_size=1 \
  --max_num_batches=50 \
  --device=DEVICE \
  --server=SERVER \
  --final_layer_on_device=N  # For mobilenet, must be >= 0 and < 14
```
