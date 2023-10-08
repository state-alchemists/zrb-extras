from zrb import runner, CmdTask
import _automate._project as _project
import _automate.zrb_extras.local as zrb_extras_local
import os

assert _project
assert zrb_extras_local

CURRENT_DIR = os.path.dirname(__file__)

playground = CmdTask(
    name='playground',
    preexec_fn=None,
    upstreams=[
        zrb_extras_local.install_zrb_extras_symlink
    ],
    retry=0,
    cwd=CURRENT_DIR,
    cmd_path=os.path.join(CURRENT_DIR, 'cmd', 'playground.sh')
)
runner.register(playground)
