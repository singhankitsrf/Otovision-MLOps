# Kubernetes deployment

The manifests demonstrate a production-style deployment surface for the
FastAPI service.

Before deployment:

1. build and push your image;
2. replace the example image reference;
3. provision a PVC or other read-only model-volume strategy;
4. place the trained checkpoint at `/models/best.pt`;
5. configure ingress, TLS and authentication outside this minimal example.

Apply:

```bash
kubectl apply -f deploy/k8s/
```

The HPA needs Metrics Server or another compatible metrics pipeline.
