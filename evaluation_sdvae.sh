CUDA_INDEX=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
GPUS=$(echo "${CUDA_INDEX}" | awk -F "," '{print NF}')
PORT=${MASTER_PORT:-29500}
echo "GPU NUM: ${GPUS}, INDEX: ${CUDA_INDEX}, MASTER PORT: ${PORT}"
args=${@:1}

torchrun --nproc_per_node=$GPUS --master_port=$PORT evaluation_sdvae.py $args