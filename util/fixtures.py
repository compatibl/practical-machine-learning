# Copyright (C) 2021-present CompatibL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import os


@pytest.fixture(scope="function")
def local_test_dir(request):
    """
    Execute test in its source directory so it can
    load data files placed next to its source.
    """

    # Change test working directory to the directory of test source
    os.chdir(request.fspath.dirname)

    # Back to the test
    yield

    # Change directory back before exiting the text
    os.chdir(request.config.invocation_dir)
