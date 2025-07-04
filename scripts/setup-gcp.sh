#!/bin/bash

# Exit on error
set -e

# Set variables
PROJECT_ID="teak-amphora-464204-a7"
REGION="us-central1"
SERVICE_ACCOUNT_NAME="constraint-lattice-sa"
REPOSITORY_NAME="constraint-lattice-repo"
CLUSTER_NAME="constraint-lattice-cluster"

# Set the project
echo "Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable \
  container.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  containerregistry.googleapis.com \
  run.googleapis.com

# Create Artifact Registry repository
echo "Creating Artifact Registry repository..."
gcloud artifacts repositories create $REPOSITORY_NAME \
  --repository-format=docker \
  --location=$REGION \
  --description="Docker repository for Constraint Lattice" || echo "Repository may already exist"

# Create a service account
echo "Creating service account..."
gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
  --display-name="Constraint Lattice Service Account" || echo "Service account may already exist"

# Assign roles to the service account
echo "Assigning roles to service account..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/container.developer"

# Create a GKE cluster
echo "Creating GKE cluster..."
gcloud container clusters create-auto $CLUSTER_NAME \
  --region=$REGION \
  --project=$PROJECT_ID

# Get credentials for the cluster
echo "Getting cluster credentials..."
gcloud container clusters get-credentials $CLUSTER_NAME \
  --region=$REGION \
  --project=$PROJECT_ID

# Create a secret for the service account
echo "Creating Kubernetes secret for service account..."
gcloud iam service-accounts keys create key.json \
  --iam-account=${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com

kubectl create secret generic gcp-key \
  --from-file=key.json \
  --dry-run=client -o yaml | kubectl apply -f -

rm key.json

echo "GCP setup completed successfully!"
echo "Next steps:"
echo "1. Add the following secrets to your GitHub repository:"
echo "   - GCP_PROJECT_ID: $PROJECT_ID"
echo "   - GCP_SA_KEY: Contents of the service account key (you can create a new one if needed)"
echo "2. Push your code to trigger the GitHub Actions workflow"
echo "3. The workflow will build and deploy your application to GKE"
