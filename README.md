# IRLAS: Inverse Reinforcement Learning for Architecture Search

PyTorch implementation of [IRLAS](https://arxiv.org/abs/1812.05285). 

If you use the code, please cite:
```bash
@inproceedings{guo2019irlas,
  title={Irlas: Inverse reinforcement learning for architecture search},
  author={Guo, Minghao and Zhong, Zhao and Wu, Wei and Lin, Dahua and Yan, Junjie},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9021--9029},
  year={2019}
}
```

## Requirements

- PyTorch 0.4.0

## Data and Model Preparation

- Download the ImageNet validation set. Put the images to `**your_dataset_path**/val` and the label file to `**your_dataset_path**/meta/val.txt`. The label file could be downloaded from [val.txt](https://drive.google.com/file/d/1VOiga8ZQa0v9x8T5_G9EVu9npy3SCU6r/view?usp=sharing).
- Download the pretrained model [IRLAS-ImageNet-mobile](https://drive.google.com/file/d/1PsylfTDKwblYsV5WYFtsYKetxSQAsFx-/view?usp=sharing) and move it to `pretrained_models`

## Test Accuracy

Set `--dataset_path=**your_dataset_path**` in `test.sh` and execute,
```bash
bash test.sh
```

The printed lines should read:
```bash
[**current time**][validate.py][line: 133][INFO]   Total params: 9.96M
[**current time**][validate.py][line:  70][INFO] Test: [0/196]   Time 11.517 (11.517)    Loss 1.1165 (1.1165)    Prec@1 72.266 (72.266)  Prec@5 92.578 (92.578)
[**current time**][validate.py][line:  70][INFO] Test: [100/196] Time 0.205 (0.574)      Loss 1.1337 (1.0932)    Prec@1 73.438 (75.174)  Prec@5 91.016 (92.168)
[**current time**][validate.py][line:  72][ INFO]  * Prec@1 75.150 Prec@5 92.090
```

## Test Inference Latency

To match the measurement of inference latency in real products, the model is converted to Caffe in `caffe_prototxt`. *Note* that all the `BatchNorm` & `Scale` layers have been merged to `Convolution` layers, which is widely used to speed up inference in real products.

Download [libnvinfer.so.4](https://drive.google.com/file/d/18IpKGEhDDYwEN8YpxDZG9ideqFJowKjX/view?usp=sharing) & [_netrt.cpython-36m-x86_64-linux-gnu.so](https://drive.google.com/file/d/1v4vcEWCX1sPIMm8vBXjkK_vlRRMy7gId/view?usp=sharing) and put them under `tools/netrt`.

The inference latency is measured on TensorRT framework. The test tools are included in `libnvinfer.so.4` and `_netrt.cpython-36m-x86_64-linux-gnu.so`. Unfortunately, these `.so` is compiled using SenseTime internal tools, which I do not have access to. However, if you happen to have all the required sources (you can check by executing `ldd libnvinfer.so.4` & `ldd _netrt.cpython-36m-x86_64-linux-gnu.so`), you can test the influence latency of the model by
```bash
bash test_time.sh
```

The last printed lines should read:
```bash
IRLAS avg cost: 9.308ms
```

The above latency is measured on TiTan Xp with 16 batch size, 224x224 input size. The latency may have a difference of around $$\pm0.5$$ms due to the fluctuation of occupancy rate or the difference of platform. When measured on 1080Ti, the latency will increase around $$1$$ms.
