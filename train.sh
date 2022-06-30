PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

python train.py --config='configs/tfnet-r50_256_20k_breast.py' --gpu-ids=0
python train.py --config='configs/tfnet-r50_256_20k_busi.py' --gpu-ids=0
python train.py --config='configs/tfnet-r50_256_20k_ddti.py' --gpu-ids=0