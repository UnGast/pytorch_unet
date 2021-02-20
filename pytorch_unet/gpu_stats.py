#
# Author : Alwyn Mathew
#
# Purpose : GPU status
#

from collections import OrderedDict
import json
import subprocess
import sys
import time
import xml.etree.ElementTree
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

print("automatically set env variable CUDA_DEVICE_ORDER to PCI_BUS_ID to set device ordering in pytorch to same ordering as nvidia-smi")

def extract(elem, tag, drop_s):
  text = elem.find(tag).text
  if drop_s not in text: raise Exception(text)
  text = text.replace(drop_s, "")
  try:
    return int(text)
  except ValueError:
    return float(text)

def get_gpu_stats(device: int):
  i = 0

  d = OrderedDict()
  d["time"] = time.time()

  cmd = ['nvidia-smi', '-i', str(device), '-q', '-x']
  cmd_out = subprocess.check_output(cmd)
  gpu = xml.etree.ElementTree.fromstring(cmd_out).find("gpu")

  util = gpu.find("utilization")
  d["gpu_util"] = extract(util, "gpu_util", "%")

  d["mem_used"] = extract(gpu.find("fb_memory_usage"), "used", "MiB")
  d["mem_used_per"] = d["mem_used"] * 100 / 11171

  return d