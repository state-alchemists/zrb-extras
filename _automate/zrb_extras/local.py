from zrb import CmdTask, runner, StrInput
from zrb.builtin.group import project_group

import os

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
RESOURCE_DIR = os.path.join(PROJECT_DIR, 'src', 'zrb-extras')
PACKAGE_DIR = os.path.join(RESOURCE_DIR, 'src')

###############################################################################
# Task Definitions
###############################################################################

prepare_zrb_extras = CmdTask(
    name='prepare-zrb-extras',
    description='Prepare venv for zrb extras',
    group=project_group,
    cwd=RESOURCE_DIR,
    cmd_path=os.path.join(CURRENT_DIR, 'cmd', 'prepare-venv.sh'),
)
runner.register(prepare_zrb_extras)

build_zrb_extras = CmdTask(
    name='build-zrb-extras',
    description='Build zrb extras',
    group=project_group,
    upstreams=[prepare_zrb_extras],
    cwd=RESOURCE_DIR,
    cmd_path=os.path.join(CURRENT_DIR, 'cmd', 'build.sh'),
)
runner.register(build_zrb_extras)

publish_zrb_extras = CmdTask(
    name='publish-zrb-extras',
    description='Publish zrb extras',
    group=project_group,
    inputs=[
        StrInput(
            name='zrb-extras-repo',
            prompt='Pypi repository for zrb extras',
            description='Pypi repository for human readalbe zrb package name',
            default='pypi',
        )
    ],
    upstreams=[build_zrb_extras],
    cwd=RESOURCE_DIR,
    cmd_path=os.path.join(CURRENT_DIR, 'cmd', 'publish.sh'),
)
runner.register(publish_zrb_extras)

install_zrb_extras_symlink = CmdTask(
    name='install-zrb-extras-symlink',
    description='Install zrb extras as symlink',
    group=project_group,
    upstreams=[build_zrb_extras],
    cwd=RESOURCE_DIR,
    cmd_path=os.path.join(CURRENT_DIR, 'cmd', 'install-symlink.sh'),
)
runner.register(install_zrb_extras_symlink)
