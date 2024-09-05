i:
	@pip install -e .

#########
### DOCKER LOCAL
#########
build_container_local:
	docker build --tag=scoreforecast:dev .

run_container_local:
	docker run -it -e PORT=8000 -p 8000:8000 scoreforecast:dev

#########
## DOCKER DEPLOYMENT
#########

# Step 1 ( ONLY FIRST TIME)
allow_docker_push:
	gcloud auth configure-docker europe-west1-docker.pkg.dev

# Step 2 ( ONLY FIRST TIME)
create_artifacts_repo:
	gcloud artifacts repositories create scorecast --repository-format=docker --location=europe-west1 --description="Repository for storing images"

### Step 3 (⚠️ M1 SPECIFICALLY)
m1_build_image_production:
	docker build --platform linux/amd64 -t europe-west1-docker.pkg.dev/flavor-forecast/scorecast/scoreforecast:dev .

## Step 4
push_image_production:
	docker push europe-west1-docker.pkg.dev/flavor-forecast/scorecast/scoreforecast:dev

# Step 5
deploy_to_cloud_run:
	gcloud run deploy --image europe-west1-docker.pkg.dev/flavor-forecast/scorecast/scoreforecast:dev --memory 2Gi --region europe-west1
