FROM postgres:latest

# copying the initialization script as written here (https://hub.docker.com/_/postgres/) in the "Initialization scripts" section
COPY backend/postgres/models/sql/create_tables.sql /docker-entrypoint-initdb.d/
