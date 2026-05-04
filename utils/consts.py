import os

DEFAULT_AUTOML_IMAGE = "registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc"
DEFAULT_AUTORAG_IMAGE = "quay.io/opendatahub/odh-autorag:odh-stable"

AUTOML_IMAGE = os.getenv("RELATED_IMAGE_MPI_AUTOML_RUNTIME", DEFAULT_AUTOML_IMAGE)
AUTORAG_IMAGE = os.getenv("RELATED_IMAGE_MPI_AUTORAG_RUNTIME", DEFAULT_AUTORAG_IMAGE)
