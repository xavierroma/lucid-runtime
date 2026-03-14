# EC2 Docker Deployment

This path deploys the coordinator on a small EC2 instance and keeps GPU execution on Modal.

After the instance has Docker installed, [`deploy/ec2/deploy.sh`](/Users/xavierroma/projects/lucid-runtime/deploy/ec2/deploy.sh) can build the coordinator image on your workstation, push it to your registry, copy the runtime files to EC2, and restart the remote container in one command.

## Target shape

- EC2 instance type: `t3.micro`
- OS: Amazon Linux 2023 (x86_64)
- Runtime: Docker Engine
- Exposed service: coordinator on TCP `8080`

`t3.micro` is suitable for the coordinator because it only runs the Rust control-plane service. Do not build the coordinator image on the instance; build and push it from another machine, then pull it on EC2.

## 1) Launch the EC2 instance

Create an EC2 `t3.micro` instance running Amazon Linux 2023.

Security group minimum:

- inbound `8080/tcp` from the IP ranges that must reach the coordinator

If you plan to use the SSH transport explicitly, also allow inbound `22/tcp` from your IP.
For the default SSM-first deploy flow, SSH ingress is not required.

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

For the default SSM-first deploy flow, the instance should also be managed by AWS Systems Manager
(SSM). Amazon Linux instances with an attached instance profile commonly already meet that bar.

## 3) Configure the runtime files on your workstation

Copy [coordinator.env.example](/Users/xavierroma/projects/lucid-runtime/deploy/ec2/coordinator.env.example) to `deploy/ec2/coordinator.env`, and copy [coordinator.models.example.json](/Users/xavierroma/projects/lucid-runtime/deploy/ec2/coordinator.models.example.json) to `deploy/ec2/coordinator.models.json`:

```bash
cp deploy/ec2/coordinator.env.example deploy/ec2/coordinator.env
cp deploy/ec2/coordinator.models.example.json deploy/ec2/coordinator.models.json
```

Coordinator runtime settings:

- `API_KEY`: bearer token for your public session API clients
- `WORKER_INTERNAL_TOKEN`: bearer token used by Modal session callbacks to hit `/internal/...`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `COORDINATOR_CALLBACK_BASE_URL`: public URL Modal can call back to
- `COORDINATOR_MODELS_FILE`: in-container path for the mounted registry file; the example uses `/app/config/coordinator.models.json`

Per-model worker targets live in `deploy/ec2/coordinator.models.json`. Each entry carries:

- the model id and display name exposed by `GET /models`
- the manifest path inside the coordinator container
- the Modal `dispatch_api` base URL, token, and worker id
- the model-specific timeout values

The example file includes `yume`, `waypoint`, and `helios`. For Waypoint and Helios, the example
uses a `900` second startup timeout because cold boots can take several minutes.

For a raw EC2 public endpoint on port `8080`, use:

```bash
COORDINATOR_CALLBACK_BASE_URL=http://<ec2-public-dns>:8080
```

Deployment settings used by `deploy/ec2/deploy.sh`:

- `COORDINATOR_IMAGE_REPOSITORY`: registry repo to push to, such as:
  - `ghcr.io/<org-or-user>/lucid-coordinator`
  - `<aws-account-id>.dkr.ecr.<region>.amazonaws.com/lucid-coordinator`
  - `docker.io/<user-or-org>/lucid-coordinator`
- `COORDINATOR_IMAGE_TAG`: optional fixed tag; if unset the script uses the current git sha, adds `-dirty` when the worktree is not clean, or falls back to a timestamp outside git
- `COORDINATOR_IMAGE`: optional full image ref if you do not want repository + tag composition
- `EC2_HOST`: public DNS name or SSH host
- `EC2_DEPLOY_TRANSPORT` (optional): `auto` (default), `ssm`, or `ssh`
- `EC2_INSTANCE_ID` (optional): explicit instance id for SSM transport
- `EC2_USER` (optional, default `ec2-user`)
- `EC2_SSH_PORT` (optional, default `22`)
- `EC2_SSH_KEY_PATH` (optional)
- `EC2_REMOTE_DIR` (optional, default `$HOME/lucid-runtime/deploy/ec2` over SSH or `/home/<EC2_USER>/lucid-runtime/deploy/ec2` over SSM)
- `EC2_MODELS_FILE` (optional, default `deploy/ec2/coordinator.models.json`)
- `AWS_PROFILE` (optional): local AWS profile used for automatic ECR login before the image push
- `EC2_DOCKER_LOGIN_COMMAND` (optional): shell command run on the instance before `docker pull`; useful for private registries
- `EC2_AWS_PROFILE` (optional): AWS profile on the instance for automatic ECR login if you are not using an instance role/default credentials

## 4) Deploy from your workstation

For this setup, log into AWS with the `lucid` profile and deploy with:

```bash
aws login --profile lucid && deploy/ec2/deploy.sh
```

For ECR, `deploy.sh` automatically logs in locally before the push and automatically logs in on
the EC2 host before `docker pull` if `EC2_DOCKER_LOGIN_COMMAND` is unset.
The script prefers SSM for managed instances and falls back to SSH only when SSM is unavailable.

The deploy script:

- builds the coordinator image for Linux x86_64 with `docker buildx`
- pushes it to `COORDINATOR_IMAGE_REPOSITORY`
- materializes `coordinator.env`, `coordinator.models.json`, and `run-coordinator.sh` on the EC2 host during deployment
- prefers SSM for managed instances and falls back to SSH otherwise
- logs into ECR locally when the image repo is ECR
- runs `EC2_DOCKER_LOGIN_COMMAND` on the instance, or auto-generates the ECR login command when the image repo is ECR
- pulls the image remotely
- replaces any existing `lucid-coordinator` container
- runs it with `--restart unless-stopped`

Useful variants:

```bash
# Push a specific tag.
deploy/ec2/deploy.sh --image-tag "$(git rev-parse --short HEAD)"

# Re-roll an already-pushed image without rebuilding locally.
deploy/ec2/deploy.sh --skip-build
```

If you only want to publish the image and skip the EC2 rollout:

```bash
deploy/ec2/deploy.sh --skip-remote
```

## 5) Verify from the instance

```bash
curl http://127.0.0.1:8080/healthz
docker logs lucid-coordinator
docker ps
```

## 6) Verify from your workstation

```bash
curl http://<ec2-public-dns>:8080/healthz
```

If that works, the coordinator is reachable for:

- `GET /models`
- `POST /sessions`
- `GET /sessions/{session_id}`
- `POST /sessions/{session_id}:end`
- Modal callbacks to `/internal/sessions/{session_id}/running`
- Modal callbacks to `/internal/sessions/{session_id}/ended`

## 7) Minimal client smoke test

```bash
curl -X POST http://<ec2-public-dns>:8080/sessions \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"helios"}'
```

Expected: `202 Accepted` with a session id and client access token.

## Demo app cross-check

If you point the demo app at this coordinator, line these up:

- coordinator `API_KEY` == demo `COORDINATOR_API_KEY` for local proxy mode or `VITE_COORDINATOR_API_KEY` for direct mode
- demo `VITE_DEFAULT_MODEL` should match one of the ids present in `GET /models`; otherwise the demo falls back to the first model in the registry
- coordinator `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` must belong to the same LiveKit project as the demo `VITE_LIVEKIT_URL`
- each registry entry `dispatch_token` must match that model worker's `MODAL_DISPATCH_TOKEN`

## Notes

- For ECR, the script logs in automatically locally and remotely. If the EC2 host does not use an
  instance role/default AWS credentials, set `EC2_AWS_PROFILE` or override with
  `EC2_DOCKER_LOGIN_COMMAND`.
- For the default SSM path, the instance must be online in Systems Manager. If it is not, either
  fix SSM or force `EC2_DEPLOY_TRANSPORT=ssh` and open port `22/tcp`.
- `t3.micro` has limited memory. Pulling and running the final image is fine; Rust compilation on-instance is the wrong tradeoff.
- If you need TLS, put the instance behind an ALB or a small reverse proxy and change `COORDINATOR_CALLBACK_BASE_URL` to `https://...`.
