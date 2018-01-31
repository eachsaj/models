#!/bin/bash
set -x

export PYTHONPATH=/home/ubuntu/src/cntk/bindings/python:/home/ubuntu/models/research:/home/ubuntu/models/research/slim

python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=models/ssd_mobilenet_v1_coco/pipeline.config \
    --checkpoint_dir=models/ssd_mobilenet_v1_coco \
    --device=0.tcp.ngrok.io:15570 \
    --server=54.200.246.175:55555 \
    --final_layer_on_device=1 \
    --eval_dir=eval
