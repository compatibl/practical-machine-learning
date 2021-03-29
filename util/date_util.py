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


class DateUtil:
    """Helper methods for working with dates."""

    @staticmethod
    def get_year_month(*, sim_month: int) -> str:
        """
        Convert 0-based sequential month index into yyyy-mm format
        using January, 2000 as origin.
        """

        # Convert simulation month to calendar month starting from year 2000
        calendar_year: int = 2000 + sim_month // 12
        calendar_month: int = 1 + sim_month % 12
        result: str = f"{calendar_year}-{str(calendar_month).zfill(2)}"
        return result

    @staticmethod
    def get_sequential_month(*, year_month: str) -> int:
        """
        Convert year-month in yyyy-mm format to 0-based sequential month index
        using January, 2000 as origin.
        """

        # Convert simulation month to calendar month starting from year 2000
        year_month_tokens = year_month.split('-')
        calendar_year: int = int(year_month_tokens[0])
        calendar_month: int = int(year_month_tokens[1])
        sim_month = (calendar_year - 2000) * 12 + calendar_month - 1
        return sim_month

    @staticmethod
    def get_lag_label(*, lag_months: int) -> str:
        """
        Return string representation of shift in months
        """
        # Create time shift label, e.g. +2y or -1m
        if lag_months > 0:
            shift_months_abs = lag_months
            shift_sign_label = "+"
        else:
            shift_months_abs = -lag_months
            shift_sign_label = "-"
        if shift_months_abs % 12 == 0:
            shift_abs_label = str(shift_months_abs // 12) + "y"
        else:
            shift_abs_label = str(shift_months_abs) + "m"

        return shift_sign_label + shift_abs_label