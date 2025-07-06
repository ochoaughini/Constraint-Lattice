# syntax=docker/dockerfile:1

# --- Stage 1: Base PHP image ---
FROM php:8.2-apache AS base

# Install system dependencies and PHP extensions (zip is common for plugins)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libzip-dev \
        unzip \
    && docker-php-ext-install zip \
    && rm -rf /var/lib/apt/lists/*

# Enable Apache mod_rewrite (commonly needed for WordPress plugins)
RUN a2enmod rewrite

# --- Stage 2: Plugin code and non-root user ---
FROM base AS plugin

# Create a non-root user for security
RUN useradd -m -d /home/wpuser wpuser

# Set working directory to the WordPress plugin directory
WORKDIR /var/www/html/wp-content/plugins/constraint-lattice-api

# Copy only the plugin code (avoid .git, .env, etc.)
COPY --link constraint-lattice-api.php ./
COPY --link readme.txt ./

# Set permissions for the plugin directory
RUN chown -R wpuser:www-data /var/www/html/wp-content/plugins/constraint-lattice-api

USER wpuser

EXPOSE 80

CMD ["apache2-foreground"]
