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

import os


class FileUtil:
    """Helper methods for working with files."""

    @staticmethod
    def get_caller_name(*, caller_file: str) -> str:
        """Get caller script name from its __file__ variable."""

        file_path, file_name_with_ext = os.path.split(caller_file)
        file_name, file_ext = os.path.splitext(file_name_with_ext)
        return file_name
