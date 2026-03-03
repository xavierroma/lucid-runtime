docker_build(
    'lucid/coordinator',
    '.',
    dockerfile='apps/coordinator/Dockerfile',
    only=['apps/coordinator'],
)

docker_build(
    'lucid/worker',
    '.',
    dockerfile='apps/worker/Dockerfile',
    only=['apps/worker'],
)

k8s_yaml(
    helm(
        'deploy/helm/lucid-runtime',
        name='lucid-runtime',
        values=['deploy/helm/lucid-runtime/values.dev.yaml'],
    )
)

k8s_resource('lucid-runtime-coordinator', port_forwards=8080)
k8s_resource('lucid-runtime-worker')
