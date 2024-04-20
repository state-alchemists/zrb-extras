"""A Kubernetes Python Pulumi program to deploy kebab-zrb-app-name"""

import os

import pulumi
from pulumi_kubernetes.helm.v3 import Chart, LocalChartOpts

NAMESPACE: str = os.getenv("NAMESPACE", "default")

postgresql = Chart(
    "postgresql",
    LocalChartOpts(
        path="./helm-charts/postgresql",
        namespace=NAMESPACE,
        values={},
        skip_await=True,
    ),
)
pulumi.export("airflow", postgresql.metadata)
