# Managed Cloud Run (v2) service for Constraint-Lattice SaaS
# -----------------------------------------------------------------------------
# Usage example:
#   module "constraint_lattice" {
#     source              = "./deployment/infra"
#     project_id          = var.project_id
#     region              = var.region
#     image_uri           = var.image_uri
#     run_service_account = var.run_service_account
#   }
# -----------------------------------------------------------------------------

terraform {
  required_version = ">= 1.6.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.19"
    }
  }
}

variable "project_id" {
  type        = string
  description = "GCP project ID"
}

variable "region" {
  type        = string
  description = "GCP region (e.g. us-central1)"
}

variable "image_uri" {
  type        = string
  description = "Container image URI hosted in Artifact Registry"
}

variable "run_service_account" {
  type        = string
  description = "Service account email used by Cloud Run revision"
}

resource "google_cloud_run_v2_service" "constraint_lattice" {
  name     = "constraint-lattice"
  location = var.region
  project  = var.project_id

  template {
    containers {
      image = var.image_uri

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }

      env {
        name  = "CLATTICE_LOG_LEVEL"
        value = "INFO"
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 10
    }

    service_account = var.run_service_account
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

output "cloud_run_url" {
  description = "Deployed Cloud Run service URL"
  value       = google_cloud_run_v2_service.constraint_lattice.uri
}
