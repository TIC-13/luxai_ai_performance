IMAGE_NAME="classification"
IMAGE_FILE="dockerfile/example.dockerfile"

echo $(pwd)
docker build -f $IMAGE_FILE -t $IMAGE_NAME .
