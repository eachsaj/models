import argparse
import sys

parser = argparse.ArgumentParser(description='Launch device.')
parser.add_argument('--server', type=str, help='server ip:host')
parser.add_argument('--device', type=str, help='device ip:host')

args = parser.parse_args()
if not args.server:
  print('Give me a server!')
  sys.exit(0)
if not args.device:
  print('Give me a device!')
  sys.exit(0)



cluster_map = {
  'server': [args.server],
  'device': [args.device],
}


import tensorflow as tf

cluster = tf.train.ClusterSpec(cluster_map)
server = tf.train.Server(cluster, job_name='device')

server.join()
