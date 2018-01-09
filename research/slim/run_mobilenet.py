#!/bin/bash

set -x
python eval_image_classifier.py \
  --alsologtostderr \
  --checkpoint_path=mobilenet-0.25/mobilenet_v1_0.25_128.ckpt \
  --dataset_name=imagenet \
  --dataset_split_name=validation \
  --model_name=mobilenet_v1_025 \
  --preprocessing_name=mobilenet_v1 \
  --batch_size=1 \
  --max_num_batches=50 \
  --redis=192.168.1.74 \
  --final_layer_on_device=MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6:0
