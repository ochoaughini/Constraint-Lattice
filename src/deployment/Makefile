# Simplified deployment helper for dual-account setup

# Usage:
#   make deploy ENV=prod IMAGE_TAG=latest

ENV ?= dev
IMAGE_TAG ?= latest

ifeq ($(ENV),prod)
	GCLOUD_CONFIG=gpu-prod
else
	GCLOUD_CONFIG=dev-cli
endif

PROJECT_ID := $(shell gcloud config configurations describe $(GCLOUD_CONFIG) --format='value(core.project)')
DEPLOYER_SA := deployer-agent@$(shell gcloud config configurations describe dev-cli --format='value(core.project)').iam.gserviceaccount.com

CONSTRAINT_IMAGE = us-central1-docker.pkg.dev/$(PROJECT_ID)/constraint-api/engine:$(IMAGE_TAG)

.PHONY: build test deploy

build:
	gcloud builds submit --config=deployment/cloudbuild.yaml \
	  --substitutions=_IMAGE=$(CONSTRAINT_IMAGE) \
	  --project=$(PROJECT_ID) \
	  --impersonate-service-account=$(DEPLOYER_SA)

# ------------------------------------------------------------
# Run unit tests locally before building container
# ------------------------------------------------------------

test:
	pip install -r requirements-dev.txt && pytest -q

deploy: build
	gcloud run deploy cl-api \
	  --image=$(CONSTRAINT_IMAGE) \
	  --project=$(PROJECT_ID) \
	  --region=us-central1 \
	  --platform=managed \
	  --impersonate-service-account=$(DEPLOYER_SA)
