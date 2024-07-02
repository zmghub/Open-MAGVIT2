#! /bin/bash
config_file=$1

export MASTER_ADDR=`echo $VC_WORKER_HOSTS | cut -d ',' -f1`
export MASTER_PORT=${2:-12688}
export WORLD_SIZE=${MA_NUM_GPUS}
export NODE_RANK=${VC_TASK_INDEX}
export OMP_NUM_THREADS=6

echo $VC_WORKER_HOSTS
echo $MASTER_ADDR
echo $MASTER_PORT
echo $WORLD_SIZE
echo $NODE_RANK

workdir=$(cd $(dirname $0); pwd)
echo "current shell dir: ${workdir}"
sudo sed -i 's/\/root\/picasso\/wyc6\/miniconda3/\/opt\/conda/g' /opt/conda/envs/svd/bin/pip
sudo sed -i 's/\/root\/picasso\/wyc6\/miniconda3/\/opt\/conda/g' /opt/conda/envs/svd/bin/pip3

/opt/conda/envs/svd/bin/pip install jsonargparse[signatures]>=4.27.7 albumentations==1.4.4 einops lpips -i http://repo.myhuaweicloud.com/repository/pypi/simple
/opt/conda/envs/svd/bin/pip install esdk-obs-python -i http://repo.myhuaweicloud.com/repository/pypi/simple
cp ${workdir}/../models/vgg16-397923af.pth /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth
mkdir -p taming/modules/autoencoder/lpips
mkdir -p /root/taming/modules/autoencoder/lpips
cp ${workdir}/taming/modules/autoencoder/lpips/vgg.pth /root/taming/modules/autoencoder/lpips/vgg.pth
cp ${workdir}/taming/modules/autoencoder/lpips/vgg.pth taming/modules/autoencoder/lpips/vgg.pth

NODE_RANK=$NODE_RANK /opt/conda/envs/svd/bin/python ${workdir}/main.py fit --config ${workdir}/configs/${config_file}