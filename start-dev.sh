docker rm tensorflow-sandbox --force
docker run --name tensorflow-sandbox -d -v$(pwd):/workspace tensorflow/tensorflow sleep infinity
docker exec -ti tensorflow-sandbox bash