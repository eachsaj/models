#!/bin/bash

set -x
python eval_image_classifier.py \
  --alsologtostderr \
  --checkpoint_path=inception-v3/inception_v3.ckpt \
  --dataset_name=imagenet \
  --dataset_split_name=validation \
  --model_name=inception_v3 \
  --batch_size=1 \
  --max_num_batches=50 \
  --redis=192.168.1.74 \
  --final_layer_on_device=InceptionV3/InceptionV3/Conv2d_2a_3x3/Relu:0
