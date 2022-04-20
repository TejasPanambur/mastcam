#!/bin/bash
#SBATCH -o results/cmdout/texture_triplet_150Reg.out
#SBATCH -e results/cmderr/texture_triplet_150Reg.err

DIR="/data/P128CURSL/train/"
ARCH="texres18"
LR=0.0001
WD=-5
K=150
WORKERS=10
BATCH=128
EXP="results/MastCam/texture/checkpoints/"
EMBPATH="results/MastCam/texture/embeddings/rawEmbeds/"
Clustering="Kmeans"
CP="results/MastCam/texture/checkpoints/checkpoint.pth.tar" 
MODE="Triplet"
WORKPATH="results/MastCam/texture/"
NITER=20
mkdir -p ${EXP}
mkdir -p ${EMBPATH}

echo "This is MastCam dataset with texture architecture, triplet loss and kmeans for clustering" > ${WORKPATH}"readme.txt"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 texture_main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
 --lr ${LR} --wd ${WD} --k ${K} --batch ${BATCH} --verbose --workers ${WORKERS} \
 --margin ${MARGIN} --mode ${MODE} --workpath ${WORKPATH} --niter ${NITER} --clustering ${Clustering}  --resume ${CP}  --epochs 100
