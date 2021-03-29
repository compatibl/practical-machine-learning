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

from dnn.short_rate_one_hot_dnn import ShortRateOneHotDnn

if __name__ == "__main__":

    # Create and train model
    dnn = ShortRateOneHotDnn()
    dnn.learning_rate = 0.1
    dnn.train_model(caller_file=__file__)

    # Use model
    dnn.run_model(caller_file=__file__, predict_country="NOR")
