import os

from cryosparc.tools import CryoSPARC
from dotenv import load_dotenv


def load_env_variables(env_file):
    load_dotenv(env_file)
    return {
        "license_id": os.getenv("CRYOSPARC_LICENSE_ID"),
        "email": os.getenv("CRYOSPARC_EMAIL"),
        "password": os.getenv("CRYOSPARC_PASSWORD"),
        "host": os.getenv("CRYOSPARC_HOST"),
        "port": int(os.getenv("CRYOSPARC_PORT")),
    }


# Initialize CryoSPARC connection
def initialize_cryosparc(env_vars):
    return CryoSPARC(
        host=env_vars["host"],
        base_port=env_vars["port"],
        license=env_vars["license_id"],
        email=env_vars["email"],
        password=env_vars["password"],
    )
