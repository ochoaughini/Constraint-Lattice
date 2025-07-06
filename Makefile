# Minimal Makefile helper for distroless build & Cloud Run deploy

PROJECT_ID ?= teak-amphora-464204-a7
REGION     ?= us-central1
REPO       ?= constraint-lattice
TAG        ?= $(shell date +%Y%m%d-%H%M%S)
IMAGE_URI  ?= $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO)/constraint-lattice:$(TAG)

.PHONY: build push deploy all

build:
	docker build -f Dockerfile.distroless -t $(IMAGE_URI) .

push:
	docker push $(IMAGE_URI)

sbom:
	@echo "Generating SBOM via Syft…"
	syft packages dir:/opt/venv -o spdx-json > sbom.json || echo "Syft not installed – skipping"
	docker cp `docker create $(IMAGE_URI)`:sbom.json sbom.json || true

# Replace IMAGE_URI placeholder in YAML and deploy declaratively
cloudrun-yaml:
	sed 's#${IMAGE_URI}#$(IMAGE_URI)#g' deployment/cloudrun/scratch-service.yaml > /tmp/cl-svc.yaml

gcmath:
	gcloud run services replace /tmp/cl-svc.yaml --region $(REGION) --project $(PROJECT_ID)

deploy: build push cloudrun-yaml gcmath
	@echo "Deployed $(IMAGE_URI) to Cloud Run."

all: deploy
