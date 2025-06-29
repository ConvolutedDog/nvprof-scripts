# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""External function interface to Nsight Compute."""
import os
import sys
import glob

from typing import Tuple

from . import nvcc


def _check_ncu_report(nc_path: str = None) -> Tuple[bool, str]:
    """Check if Nsight Compute's Python utilities (ncu_report.py) exist and
    return the module path.

    Parameters
    ----------
    nc_path : str, optional
        Path to Nsight Compute root directory (e.g., "$CUDA_PATH/nsight-compute-2024.1.1").

    Returns
    -------
    exists : bool
        True if extras/python/ncu_report.py exists in nc_path.
    module_path : str
        Path to the directory containing ncu_report.py (i.e., "nc_path/extras/python"),
        suitable for `sys.path.append()`.
    """
    module_path = os.path.join(nc_path, "extras", "python")
    ncu_report_path = os.path.join(module_path, "ncu_report.py")
    return os.path.exists(ncu_report_path), module_path


def find_ncu_report_path(nsight_compute_dir: str = None) -> Tuple[str, str]:
    """Utility function to find the path to Nsight Compute's Python utilities
    (ncu_report.py) and the `ncu` executable.

    The returned moudule path can be directly added to `sys.path` to import
    `ncu_report`.

    Search Order:
        1. User-provided path (if specified).
        2. Default CUDA installation directory.

    Parameters
    ----------
    nsight_compute_dir : str, optional
        The Nsight Compute path in str, like "/usr/local/cuda/nsight-compute-*".

    Returns
    -------
    module_path : str
        Path to extras/python under Nsight Compute, ready for `sys.path.append()`.
    ncu_path: str
        Path to ncu executable.
    """

    # TODO (ConvolutedDog) Nsight Compute CLI (ncu) version is decoupled
    # from the CUDA toolkit, so we can use a newer or older version of
    # ncu independently.

    if nsight_compute_dir is not None:
        exist, module_path = _check_ncu_report(nsight_compute_dir)
        print(exist, module_path)
        if exist:
            ncu_path = os.path.join(module_path, "..", "..", "ncu")
            return module_path, ncu_path
        else:
            raise RuntimeError(
                "Cannot find path of ncu_report.py in the provided Nsight "
                f"Compute path '{nsight_compute_dir}', please make sure "
                "the path is correct or update your cuda version."
            )

    cuda_path = nvcc.find_cuda_path()
    nsight_compute_dirs = sorted(
        glob.glob(os.path.join(cuda_path, "nsight-compute-*")), reverse=True
    )  # Try newest first
    if not nsight_compute_dirs:
        raise RuntimeError(
            "Cannot find path of Nsight Compute in cuda path "
            f"'{cuda_path}', please update your cuda version."
        )

    for path in nsight_compute_dirs:
        exist, module_path = _check_ncu_report(path)
        if exist:
            ncu_path = os.path.join(module_path, "..", "..", "ncu")
            return module_path, ncu_path

    raise RuntimeError(
        "Cannot find path of ncu_report.py in the the default cuda path "
        f"'{cuda_path}', please make sure the path is correct or update "
        "your cuda version."
    )
