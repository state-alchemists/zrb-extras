from zrb import Env

###############################################################################
# Env Definitions
###############################################################################

image_env = Env(
    name='IMAGE',
    os_name='CONTAINER_ZRB_ENV_PREFIX_IMAGE',
    default='stalchmst/moku:1.0.0'
)
