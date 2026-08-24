# Security Policy

This portfolio project must not contain patient-identifiable information,
credentials, private keys, access tokens, or production secrets.

## Reporting

If you find a security issue, do not place sensitive details in a public issue.
Contact the repository owner privately.

## Deployment principles

- run the container as a non-root user;
- mount model files read-only;
- store secrets in the target platform secret manager;
- use authentication and TLS in any real deployment;
- do not log raw clinical images by default;
- apply least-privilege RBAC in Kubernetes.
