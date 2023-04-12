docker rm tensorflow-titanic-example --force
docker image build -t tensorflow-titanic-example .
docker run -ti --name tensorflow-titanic-example tensorflow-titanic-example
