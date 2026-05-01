# syntax=docker/dockerfile:1.7
# Dockerfile for the RiskML capstone Streamlit dashboard.
# Single-stage build, python:3.11-slim base, non-root runtime user.
# Designed for Azure Container Apps deployment (Phase 3) and local
# verification (Phase 2).

# Base image: official Python 3.11 slim variant. ~50 MB; includes pip,
# excludes most system tooling (we add only what we need).
FROM python:3.11-slim

# Build-time argument for app version tag. Surfaces in image labels for
# traceability when ACA rolls forward and back across versions.
ARG APP_VERSION=phase2-dev

# OCI image labels — visible via `docker inspect` and in ACR / ACA UI.
LABEL org.opencontainers.image.title="riskml-dashboard" \
    org.opencontainers.image.description="Causal-aware ML risk forecasting dashboard" \
    org.opencontainers.image.source="https://github.com/stevearchuleta/riskml-capstone" \
    org.opencontainers.image.version="${APP_VERSION}"

# Python runtime configuration.
# PYTHONDONTWRITEBYTECODE=1 prevents .pyc files inside the container.
# PYTHONUNBUFFERED=1 forces stdout/stderr to be unbuffered for live logs.
# PIP_NO_CACHE_DIR=1 prevents pip from caching wheels (smaller image).
# PIP_DISABLE_PIP_VERSION_CHECK=1 silences the upgrade nag at install time.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Working directory inside the container.
WORKDIR /app

# OS-level dependencies. python:3.11-slim is intentionally minimal, so
# scientific Python packages occasionally need a couple of build helpers.
# curl is included for the HEALTHCHECK below.
# After install, clean apt lists to keep the image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependency manifest AND the package source first. Both are
# needed for `pip install .` to succeed because pyproject.toml declares
# riskml as an installable package — pip needs the source present at
# install time, not just the manifest.
# This pattern still preserves the build cache: the slow pip install
# layer only invalidates when deps or package source change, not when
# dashboard code (app/) or artifacts (reports/, data/) change.
COPY pyproject.toml README.md ./
COPY riskml/ ./riskml/

# Install Python dependencies.
# `pip install .` reads pyproject.toml, resolves dependencies, and
# installs the riskml package into the system site-packages.
# Pip itself is pinned to a known-good major-version range for
# reproducible builds across rebuilds and CI runs.
# We intentionally avoid editable installs (-e) here: editable installs
# create a development shim that's wrong for a frozen production image.
# Plain `pip install .` (no [dev] extras) excludes pytest, ruff, and
# other dev-only tools from the production image.
RUN pip install --upgrade "pip>=24.0,<26.0" && \
    pip install .

# Copy the application code and read-only artifacts.
# .dockerignore filters out notebooks/, tests/, .git/, caches, and
# IDE noise. What lands here is: app/, riskml/, data/processed/,
# reports/, README.md, and the pyproject.toml already present.
COPY . .

# Create a non-root user and group, then chown /app so the runtime user
# can read everything. This drops root privileges for the dashboard
# process, a basic but expected production hardening.
RUN groupadd --system --gid 1001 appuser && \
    useradd --system --uid 1001 --gid appuser --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Streamlit listens on 8501. EXPOSE is documentation for Docker tooling
# and Azure Container Apps; it does not actually publish the port.
EXPOSE 8501

# HEALTHCHECK: Streamlit exposes /_stcore/health for liveness probes.
# Useful for local container validation during Phase 2.
# Azure Container Apps will configure its own startup, liveness, and
# readiness probes at the ACA layer in Phase 3 — this Dockerfile
# instruction does not replace those.
# 30s interval, 5s timeout, 3 consecutive failures = unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# CMD runs the Streamlit server bound to 0.0.0.0 (all interfaces inside
# the container) so the host port mapping can reach it. Binding to
# 127.0.0.1 here is the single most common Docker-Streamlit bug.
# --server.headless=true disables the "open browser" prompt that
# Streamlit otherwise runs at startup (irrelevant in a container).
CMD ["streamlit", "run", "app/streamlit_dashboard.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]