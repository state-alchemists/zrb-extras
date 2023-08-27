from zrb import CmdTask, runner
from zrb.builtin.group import project_group
from .container import start_snake_zrb_app_name_container

###############################################################################
# Task Definitions
###############################################################################

start_snake_zrb_app_name = CmdTask(
    icon='🚤',
    name='start-kebab-zrb-app-name',
    description='Start human readable zrb app name',
    group=project_group,
    upstreams=[start_snake_zrb_app_name_container],
    cmd='Starting kebab-zrb-app-name as containers'
)
runner.register(start_snake_zrb_app_name)
