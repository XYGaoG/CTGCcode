device=-1

python 1preprocess.py --gpu $device --dataset cora 
python 1preprocess.py --gpu $device --dataset citeseer 
python 1preprocess.py --gpu $device --dataset ogbn-arxiv 
python 1preprocess.py --gpu $device --dataset reddit 
