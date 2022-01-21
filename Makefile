.PHONY: docker-build
docker-build:
	docker build -t anime2sketch .

.PHONY: docker-run
docker-run:
	docker run -it --rm --gpus all -v `pwd`:/workspace -v `pwd`/images/input:/input -v `pwd`/images/output:/output anime2sketch