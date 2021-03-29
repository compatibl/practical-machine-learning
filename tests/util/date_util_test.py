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

from util.date_util import DateUtil


class DateUtilTest:
    """Tests for DateUtil class."""

    def get_year_month_test(self):
        """Test for get_year_month method."""

        assert DateUtil.get_year_month(sim_month=0) == "2000-01"
        assert DateUtil.get_year_month(sim_month=11) == "2000-12"
        assert DateUtil.get_year_month(sim_month=120) == "2010-01"

    def get_sequential_month_test(self):
        """Test for get_year_month method."""

        assert DateUtil.get_sequential_month(year_month="2000-01") == 0
        assert DateUtil.get_sequential_month(year_month="2000-12") == 11
        assert DateUtil.get_sequential_month(year_month="2010-01") == 120

    def get_lag_label_test(self):
        """Test for get_lag_label method."""

        assert DateUtil.get_lag_label(lag_months=6) == "+6m"
        assert DateUtil.get_lag_label(lag_months=-6) == "-6m"
        assert DateUtil.get_lag_label(lag_months=12) == "+1y"
        assert DateUtil.get_lag_label(lag_months=-12) == "-1y"
        assert DateUtil.get_lag_label(lag_months=24) == "+2y"
        assert DateUtil.get_lag_label(lag_months=-24) == "-2y"


if __name__ == "__main__":
    pytest.main([__file__])
