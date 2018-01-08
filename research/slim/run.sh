#!/bin/bash

set -x
for i in `seq -1 13`;
do
	python eval_image_classifier.py \
		--alsologtostderr \
		--checkpoint_path=mobilenet-0.25/mobilenet_v1_0.25_128.ckpt \
		--dataset_name=imagenet \
		--dataset_split_name=validation \
		--model_name=mobilenet_v1_025 \
		--preprocessing_name=mobilenet_v1 \
		--batch_size=50 \
		--max_num_batches=3 \
		--device=192.168.1.10:55555 \
		--server=192.168.1.74:55555 \
		--final_layer_on_device=$i
done    
