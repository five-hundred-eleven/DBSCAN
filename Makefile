build:
	docker build --tag dbscan:latest .

up: build
	docker run --detach -e PORT=51000 -p 51000:51000 --name dbscan_test dbscan:latest

deploy: build
	docker tag dbscan:latest registry.heroku.com/ecowley-dbscan/web
	docker push registry.heroku.com/ecowley-dbscan/web
	heroku container:release web --app ecowley-dbscan

clean:
	docker kill dbscan_test
