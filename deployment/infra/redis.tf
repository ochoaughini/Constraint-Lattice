resource "google_redis_instance" "constraint_cache" {
  name           = "constraint-cache"
  tier           = "STANDARD_HA"
  memory_size_gb = 1
  region         = var.region
  project        = var.project_id
}
