# Lucid Runtime Monorepo

This repository is structured as a monorepo for the V1 runtime in `spec.md`.

## Layout

- `apps/coordinator`: Rust CPU coordinator service.
- `apps/worker`: Python GPU worker service.
- `packages/contracts`: Shared API contracts (OpenAPI, schemas).
- `deploy/helm/lucid-runtime`: Helm chart for coordinator + worker deployment.
- `Tiltfile`: Local Kubernetes development entrypoint.
- `scripts`: Utility scripts.

## Tooling choices

- Deploy definition: Helm
- Dev workflow: Tilt

## Local dev (Tilt)

1. Start a local Kubernetes cluster (for example: `minikube`, `kind`, or Docker Desktop Kubernetes).
2. Run:
   ```bash
   tilt up
   ```

## Deploy (Helm)

```bash
helm upgrade --install lucid-runtime deploy/helm/lucid-runtime \
  --namespace lucid-runtime \
  --create-namespace
```

## Notes

- V1 scope is single GPU + single worker.
- The initial app code is scaffold-only; implement runtime behavior against `spec.md`.
