"""Deprecated. Use config.CryoSPARCConnectionConfig instead.

Kept only for backward compatibility with any external code that still
imports load_env_variables / initialize_cryosparc from here.
"""

from __future__ import annotations

from cryosparc.tools import CryoSPARC

from config import CryoSPARCConnectionConfig


def load_env_variables(env_file):
    values = CryoSPARCConnectionConfig(env_file=env_file).load()
    return {
        "license_id": values["CRYOSPARC_LICENSE_ID"],
        "email": values["CRYOSPARC_EMAIL"],
        "password": values["CRYOSPARC_PASSWORD"],
        "host": values["CRYOSPARC_HOST"],
        "port": int(values["CRYOSPARC_PORT"]),
    }


def initialize_cryosparc(env_vars):
    return CryoSPARC(
        host=env_vars["host"],
        base_port=env_vars["port"],
        license=env_vars["license_id"],
        email=env_vars["email"],
        password=env_vars["password"],
    )
