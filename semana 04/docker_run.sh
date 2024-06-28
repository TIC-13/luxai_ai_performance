
DOCKER_IMAGE="classification"

CONTAINER_NAME="classification-ct"

WORK_DIR="/$DOCKER_IMAGE" 

DATASET_PATH="`pwd`/../data"

PORT=8888:8888

if [ -z $1 ]
then
    echo "using default port ($PORT)"
else
    PORT=$1
    echo "Using port $PORT"
fi

sudo docker run -it -p $PORT --user root --rm --gpus all \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --gpus all \
    --shm-size 8G \
    --workdir=$WORK_DIR \
    --name=$CONTAINER_NAME \
    --privileged \
    --oom-kill-disable \
    --volume="`pwd`:$WORK_DIR" \
    --volume="$DATASET_PATH:$WORK_DIR/../data:ro" \
    --volume="/usr/local/cuda/bin:/usr/local/cuda/bin:ro" \
    --volume="/usr/local/cuda/lib64:/usr/local/cuda/lib64:ro" \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
    $DOCKER_IMAGE

exit $?
