#!/usr/bin/env bash
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
set -v
set -x

if [[ -z $PYTHON_NAME ]]; then
  # the default option -- note that the minimal version of
  # python that is suitable for this repo is python3.7,
  # whereas the default python3 may point to python3.6
  PYTHON_NAME=python3
fi

PYTHON_VERSION=$($PYTHON_NAME --version | sed -e "s/^Python \([0-9]\.[0-9]\)\..*/\1/") || exit 1
if [[ $PYTHON_VERSION != "3.7" && $PYTHON_VERSION != "3.8" && $PYTHON_VERSION != "3.9" ]]; then
  echo "Wrong version of python: '$PYTHON_VERSION'"
  exit 1
fi

# Get CUDA version.
CUDA_HOME_CANDIDATE=/usr/local/cuda
if [ -z "${CUDA_HOME}" ] && [ -d ${CUDA_HOME_CANDIDATE} ]; then
  echo "Exporting CUDA_HOME as ${CUDA_HOME_CANDIDATE}"
  export CUDA_HOME=${CUDA_HOME_CANDIDATE}
fi

if [ -e "$CUDA_HOME" ]; then
  if [ -e "$CUDA_HOME/version.txt" ]; then
    # Get CUDA version from version.txt file.
    CUDA_VERSION=$(cat $CUDA_HOME/version.txt | sed -e "s/^.*CUDA Version *//" -e "s/ .*//")
  else
    # Get CUDA version from directory name.
    CUDA_HOME_DIR=$(readlink -f $CUDA_HOME)
    CUDA_HOME_DIR=$(basename "$CUDA_HOME_DIR")
    CUDA_VERSION=$(echo "$CUDA_HOME_DIR" | cut -d "-" -f 2)
  fi
fi

# install PyTorch and MMCV.
export TORCH_VERSION=1.8.2
export TORCHVISION_VERSION=0.9.2
export MMCV_VERSION=1.3.14

if [[ -z ${CUDA_VERSION} ]]; then
  echo "CUDA was not found, installing dependencies in CPU-only mode. If you want to use CUDA, set CUDA_HOME and CUDA_VERSION beforehand."
else
  # Remove dots from CUDA version string, if any.
  CUDA_VERSION_CODE=$(echo "${CUDA_VERSION}" | sed -e "s/\.//" -e "s/\(...\).*/\1/")
  echo "Using CUDA_VERSION ${CUDA_VERSION}"
  if [[ "${CUDA_VERSION_CODE}" != "111" ]] && [[ "${CUDA_VERSION_CODE}" != "102" ]] ; then
    echo "CUDA version must be either 11.1 or 10.2"
    exit 1
  fi
  echo "export CUDA_HOME=${CUDA_HOME}"
fi

CONSTRAINTS_FILE=$(mktemp)
cat constraints.txt >> "${CONSTRAINTS_FILE}"
export PIP_CONSTRAINT=${CONSTRAINTS_FILE}

# Newer versions of pip have troubles with NNCF installation from the repo commit.
pip install pip==21.2.1 || exit 1
pip install wheel || exit 1
pip install --upgrade setuptools || exit 1

if [[ -z $CUDA_VERSION_CODE ]]; then
  export TORCH_VERSION=${TORCH_VERSION}+cpu
  export TORCHVISION_VERSION=${TORCHVISION_VERSION}+cpu
else
  export TORCH_VERSION=${TORCH_VERSION}+cu${CUDA_VERSION_CODE}
  export TORCHVISION_VERSION=${TORCHVISION_VERSION}+cu${CUDA_VERSION_CODE}
fi

# Install pytorch
echo torch=="${TORCH_VERSION}" >> "${CONSTRAINTS_FILE}"
echo torchvision=="${TORCHVISION_VERSION}" >> "${CONSTRAINTS_FILE}"
pip install torch=="${TORCH_VERSION}" torchvision=="${TORCHVISION_VERSION}" -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html || exit 1

# Install mmcv
pip install --no-cache-dir mmcv-full==${MMCV_VERSION} || exit 1

# Install Base requirements
pip install -r requirements/base.txt || exit 1
pip install -r requirements/openvino.txt || exit 1

# Install backends
pip install numpy==1.21.4
pip install --no-cache-dir --no-binary=mmpycocotools mmpycocotools || exit 1
pip install -e .
# pip install mmdet@git+https://github.com/openvinotoolkit/mmdetection@ote || exit 1
# pip install mpa@git+https://github.com/openvinotoolkit/model_preparation_algorithm@develop || exit 1


# Build NNCF extensions
echo "Build NNCF extensions ..."
python -c "import nncf"
