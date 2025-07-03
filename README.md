## Local Development Setup with Docker Compose

This project can be easily run locally using Docker Compose. Ensure you have Docker and Docker Compose installed.

### Prerequisites

*   Docker Desktop (or Docker Engine + Docker Compose)
*   A Google Cloud project with billing enabled and the Artifact Registry API enabled if you plan to push images to GCP.
*   Your GitHub repository connected to Workload Identity Federation (for CI/CD).

### Local Development Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Constraint-Lattice.git
    cd Constraint-Lattice
    ```

2.  **Set up the environment:**
    *   Create a `.env` file in the root directory of the project.
    *   Copy the contents of the example `.env` file provided in the project (or create it manually) and fill in your desired database credentials and API keys. Ensure `ENABLE_SAAS_FEATURES=true` is set for the backend.
    *   Example `.env` content:
        ```dotenv
        # FastAPI Backend
        ENABLE_SAAS_FEATURES=true
        CLATTICE_API_URL=http://localhost:8000
        CLATTICE_API_KEY= # Leave blank or set if needed for local testing

        # Redis
        REDIS_URL=redis://redis:6379/0

        # WordPress Database
        MYSQL_ROOT_PASSWORD=your_mysql_root_password
        MYSQL_USER=wordpress_user
        MYSQL_PASSWORD=wordpress_password
        MYSQL_DATABASE=wordpress_db
        ```
        *Remember to replace placeholder passwords and usernames.*

3.  **Build and run the containers:**
    ```bash
    docker-compose up --build
    ```
    This command will:
    *   Build the Docker images for your FastAPI backend and the WordPress plugin.
    *   Start the FastAPI backend, Redis, MySQL, and WordPress containers.

4.  **Access the services:**
    *   **FastAPI Backend:** Your API should be accessible at `http://localhost:8000`. You can test the health endpoint with `curl http://localhost:8000/health`.
    *   **WordPress:** Your WordPress site should be accessible at `http://localhost:8080`. Follow the on-screen instructions to complete the WordPress installation. Once installed, activate the "Constraint Lattice API" plugin and test the `[clattice_demo]` shortcode on a page to see it interact with your local backend.

5.  **Stopping the containers:**
    To stop the running containers, press `Ctrl+C` in the terminal where `docker-compose up` is running, or run:
    ```bash
    docker-compose down
    ```

---

With these files in place, you should be able to run your entire application stack locally using a single command. Please ensure all these files are created in their respective locations within your project.