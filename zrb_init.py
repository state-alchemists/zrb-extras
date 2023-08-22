from zrb import runner, CmdTask
import _automate._project as _project
import _automate.zrb_extras.local as zrb_extras_local

assert _project
assert zrb_extras_local

playground = CmdTask(
    name='playground',
    preexec_fn=None,
    upstreams=[
        zrb_extras_local.install_zrb_extras_symlink
    ],
    cmd=[
        'sudo rm -Rf playground',
        'cp -R playground-template playground',
        'cd playground',
        './seed.sh',
    ]
)
runner.register(playground)

