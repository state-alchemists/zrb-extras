from zrb import DockerComposeTask, Env, EnvFile, runner
from zrb.builtin.group import project_group
from .image import image_env
from ._common import RESOURCE_DIR, local_input, host_input
from .checker import snake_zrb_app_name_container_checker
import os

###############################################################################
# Env File Definitions
###############################################################################

compose_env_file = EnvFile(
    env_file=os.path.join(RESOURCE_DIR, 'docker-compose.env'),
    prefix='CONTAINER_ZRB_ENV_PREFIX'
)

###############################################################################
# Env Definitions
###############################################################################

host_port_env = Env(
    name='HOST_PORT',
    os_name='CONTAINER_ZRB_ENV_PREFIX_HOST_PORT',
    default='zrbAppPort'
)

###############################################################################
# Task Definitions
###############################################################################

remove_snake_zrb_app_name_container = DockerComposeTask(
    icon='💨',
    name='remove-kebab-zrb-app-name-container',
    description='Remove human readable zrb app name container',
    group=project_group,
    cwd=RESOURCE_DIR,
    compose_cmd='down',
    compose_env_prefix='CONTAINER_ZRB_ENV_PREFIX',
    env_files=[compose_env_file],
    envs=[
        image_env,
        host_port_env,
    ],
)
runner.register(remove_snake_zrb_app_name_container)

stop_snake_zrb_app_name_container = DockerComposeTask(
    icon='⛔',
    name='stop-kebab-zrb-app-name-container',
    description='Stop human readable zrb app name container',
    group=project_group,
    cwd=RESOURCE_DIR,
    compose_cmd='stop',
    compose_env_prefix='CONTAINER_ZRB_ENV_PREFIX',
    env_files=[compose_env_file],
    envs=[
        image_env,
        host_port_env,
    ],
)
runner.register(stop_snake_zrb_app_name_container)

init_snake_zrb_app_name_container = DockerComposeTask(
    icon='🔥',
    name='init-kebab-zrb-app-name-container',
    inputs=[
        local_input,
        host_input,
    ],
    skip_execution='{{not input.local_snake_zrb_app_name}}',
    upstreams=[
        remove_snake_zrb_app_name_container,
    ],
    cwd=RESOURCE_DIR,
    compose_cmd='up',
    compose_flags=['-d'],
    compose_env_prefix='CONTAINER_ZRB_ENV_PREFIX',
    env_files=[compose_env_file],
    envs=[
        image_env,
        host_port_env,
    ],
)

start_snake_zrb_app_name_container = DockerComposeTask(
    icon='🐳',
    name='start-kebab-zrb-app-name-container',
    description='Start human readable zrb app name container',
    group=project_group,
    inputs=[
        local_input,
        host_input,
    ],
    skip_execution='{{not input.local_snake_zrb_app_name}}',
    upstreams=[init_snake_zrb_app_name_container],
    cwd=RESOURCE_DIR,
    compose_cmd='logs',
    compose_flags=['-f'],
    compose_env_prefix='CONTAINER_ZRB_ENV_PREFIX',
    env_files=[compose_env_file],
    envs=[
        image_env,
        host_port_env,
    ],
    checkers=[
        snake_zrb_app_name_container_checker,
    ]
)
runner.register(start_snake_zrb_app_name_container)
