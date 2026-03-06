# EC2 Docker Deployment

This path deploys the coordinator on a small EC2 instance and keeps GPU execution on Modal.

## Target shape

- EC2 instance type: `t3.micro`
- OS: Amazon Linux 2023 (x86_64)
- Runtime: Docker Engine
- Exposed service: coordinator on TCP `8080`

`t3.micro` is suitable for the coordinator because it only runs the Rust control-plane service. Do not build the coordinator image on the instance; build and push it from another machine, then pull it on EC2.

## 1) Launch the EC2 instance

Create an EC2 `t3.micro` instance running Amazon Linux 2023.

Security group minimum:

- inbound `22/tcp` from your IP
- inbound `8080/tcp` from the IP ranges that must reach the coordinator

For a quick test, `8080/tcp` can be open publicly. For anything beyond a short-lived test, restrict that rule and front the instance with a proper HTTPS endpoint.

## 2) Install Docker on the instance

```bash
sudo dnf update -y
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user
newgrp docker
docker version
```

## 3) Build and push the image from your workstation

Build for Linux x86_64, then push to any registry the instance can pull from.

```bash
docker buildx build \
  --platform linux/amd64 \
  -f apps/coordinator/Dockerfile \
  -t <registry>/lucid-coordinator:<tag> \
  --push \
  .
```

Examples of `<registry>`:

- `ghcr.io/<org-or-user>`
- `<aws-account-id>.dkr.ecr.<region>.amazonaws.com`
- `docker.io/<user-or-org>`

## 4) Copy env template and fill it in

On the instance:

```bash
mkdir -p ~/lucid-runtime/deploy/ec2
```

Copy [coordinator.env.example](/Users/xavierroma/projects/lucid-runtime/deploy/ec2/coordinator.env.example) to `~/lucid-runtime/deploy/ec2/coordinator.env`, then set:

- `API_KEY`: bearer token for your public session API clients
- `WORKER_INTERNAL_TOKEN`: bearer token used by Modal session callbacks to hit `/internal/...`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `MODAL_DISPATCH_BASE_URL`: Modal `dispatch_api` base URL
- `MODAL_DISPATCH_TOKEN`: same token configured for the Modal app
- `COORDINATOR_CALLBACK_BASE_URL`: public URL Modal can call back to

For a raw EC2 public endpoint on port `8080`, use:

```bash
COORDINATOR_CALLBACK_BASE_URL=http://<ec2-public-dns>:8080
```

## 5) Pull and run the coordinator

Set the image reference, then start the container with the helper script.

```bash
export COORDINATOR_IMAGE=<registry>/lucid-coordinator:<tag>
bash deploy/ec2/run-coordinator.sh deploy/ec2/coordinator.env
```

The helper script:

- pulls the image
- replaces any existing `lucid-coordinator` container
- runs it with `--restart unless-stopped`

## 6) Verify from the instance

```bash
curl http://127.0.0.1:8080/healthz
docker logs lucid-coordinator
docker ps
```

## 7) Verify from your workstation

```bash
curl http://<ec2-public-dns>:8080/healthz
```

If that works, the coordinator is reachable for:

- `POST /v1/sessions`
- `GET /v1/sessions/{session_id}`
- `POST /v1/sessions/{session_id}:end`
- Modal callbacks to `/internal/v1/sessions/{session_id}/running`
- Modal callbacks to `/internal/v1/sessions/{session_id}/ended`

## 8) Minimal client smoke test

```bash
curl -X POST http://<ec2-public-dns>:8080/v1/sessions \
  -H "Authorization: Bearer <API_KEY>"
```

Expected: `202 Accepted` with a session id and client access token.

## Notes

- If you use ECR, authenticate the instance with `aws ecr get-login-password | docker login ...` before pulling.
- `t3.micro` has limited memory. Pulling and running the final image is fine; Rust compilation on-instance is the wrong tradeoff.
- If you need TLS, put the instance behind an ALB or a small reverse proxy and change `COORDINATOR_CALLBACK_BASE_URL` to `https://...`.
