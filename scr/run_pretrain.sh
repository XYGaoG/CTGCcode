device=0

python 2pretrain.py --gpu $device --dataset cora
python 2pretrain.py --gpu $device --dataset citeseer
python 2pretrain.py --gpu $device --dataset ogbn-arxiv
python 2pretrain.py --gpu $device --dataset reddit
