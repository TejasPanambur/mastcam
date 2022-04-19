#!/bin/bash
#SBATCH -o results/cmdout/texture_triplet_150Reg.out
#SBATCH -e results/cmderr/texture_triplet_150Reg.err

DIR="/local/P128CURSL/"
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
#"TripletLossXBM""MSloss" 
#GTOS39 GTOSmob31 Minc23 dtd47
MARGIN=0.05
RAD=0.8
WORKPATH="results/MastCam/texture/"
NITER=20
mkdir -p ${EXP}
mkdir -p ${EMBPATH}

echo "This is MastCam dataset with texture architecture, triplet loss and kmeans for clustering" > ${WORKPATH}"readme.txt"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 texture_main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
 --lr ${LR} --wd ${WD} --k ${K} --batch ${BATCH} --verbose --workers ${WORKERS} \
 --margin ${MARGIN} --mode ${MODE} --workpath ${WORKPATH} --niter ${NITER} --EC ${EC} --clustering ${Clustering} --rad ${RAD} --resume ${CP} --st_xbm ${ST_XBM} --epochs 100
