import os

from zrb import Env, EnvFile

from .._constant import RESOURCE_DIR

compose_env_file = EnvFile(
    path=os.path.join(RESOURCE_DIR, "docker-compose.env"),
    prefix="CONTAINER_ZRB_ENV_PREFIX",
)

airflow_webserver_port_env = Env(
    name="AIRFLOW_WEBSERVER_PORT",
    os_name="CONTAINER_ZRB_ENV_PREFIX_AIRFLOW_WEBSERVER_PORT",
    default="zrbAppHttpPort",
)
