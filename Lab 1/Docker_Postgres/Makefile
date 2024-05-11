up:
	docker-compose -f docker-compose.yaml up --force-recreate

rm:
	docker-compose stop \
	&& docker-compose rm \
	&& rmdir data/ \
	&& rmdir servers.json/