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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import attr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from util.file_util import FileUtil
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


@attr.s(slots=True, auto_attribs=True)
class ShortRateDnn:
    """
    Deep neural network for performing regression as a function
    of short rate only.
    """

    input_file: str = attr.ib(default=None, kw_only=True)
    """
    Determines the file from which the data is taken.
    The file name format is {caller_name}.{feature}.csv

    The plot includes every column for every feature.
    """

    learning_rate: float = attr.ib(default=None, kw_only=True)
    """
    Learning rate used for the optimizer.
    """

    skip_samples: int = attr.ib(default=None, kw_only=True)
    """
    Optional number of samples to skip in input_file.
    """

    take_samples: int = attr.ib(default=None, kw_only=True)
    """
    Optional number of samples to take in input_file,
    after skipping skip_samples.
    """

    __input_dataset: pd.DataFrame = attr.ib(default=None, kw_only=True)
    """
    Complete input dataset (without applying skip_samples and take_samples).
    
    This object is only used for comparison.
    """

    __train_dataset: pd.DataFrame = attr.ib(default=None, kw_only=True)
    """
    Train dataset for comparison to model results.
    """

    __model: keras.Sequential = attr.ib(default=None, kw_only=True)
    """
    TF model object.
    """

    __short_rate_feature = "short_rate(t)"
    """Regression is performed with respect to this feature."""

    __lag_short_rate_feature = "short_rate(t+5y)"
    """Regression is performed to find mean of this feature."""

    def train_model(self, *, caller_file: str):
        """
        Perform model training on data from input_file.

        Pass __file__ variable of the caller script as caller_file
        parameter. It will be used as output file prefix.
        """

        # Make numpy printouts easier to read
        np.set_printoptions(precision=3, suppress=True)

        # Set random seed for both Python and TF
        # to make the results reproducible
        seed = 0
        np.random.RandomState(seed)
        tf.random.set_seed(seed)

        # Input file has the same name as the caller script
        # and csv extension, unless specified otherwise.
        if self.input_file is None:
            input_file = f"{FileUtil.get_caller_name(caller_file=caller_file)}.csv"
        else:
            input_file = self.input_file

        # Read the dataset
        self.__input_dataset = pd.read_csv(input_file)

        # Skip the specified number of samples
        if self.skip_samples is not None:
            self.__input_dataset = self.__input_dataset.tail(self.skip_samples)

        # Then take the specified number of samples
        if self.take_samples is not None:
            self.__input_dataset = self.__input_dataset.head(self.take_samples)

        # Convert LOCATION column to one-hot encoding to avoid
        # model bias due to the currency order in sample.
        self.__train_dataset = self.__input_dataset.copy()

        # Split features from labels

        # Remove target series from train dataset and save it to a separate variable
        target_series = self.__train_dataset.pop(self.__lag_short_rate_feature)

        # Create a normalizer layer and adapt it to data
        short_rate = np.array(self.__train_dataset[self.__short_rate_feature])
        normalizer = preprocessing.Normalization(input_shape=[1, ])
        normalizer.adapt(short_rate)

        # Create DNN model (not yet deep in this example)
        self.__model = keras.Sequential([
            normalizer,
            layers.Dense(64, activation='sigmoid'),
            layers.Dense(1)
        ])

        # Compile the model
        self.__model.compile(
            loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(self.learning_rate)
        )

        # Print model summary
        print(self.__model.summary())

        # Perform training and save training history in a variable
        # Fit is performed by using validation_split fraction of the data
        # to train and the remaining data to minimize.
        training_history = self.__model.fit(
            self.__train_dataset[self.__short_rate_feature],
            target_series,
            validation_split=0.5,
            verbose=0,
            epochs=100)

    def run_model(self, *, caller_file: str) -> None:
        """
        Run model on the specified data.

        Pass __file__ variable of the caller script as caller_file
        parameter. It will be used as output file prefix.
        """

        short_rate_grid = tf.linspace(-5, 25, 31, True)
        test_predictions = self.__model.predict(short_rate_grid)

        # Data only for all countries
        all_args = self.__input_dataset[self.__short_rate_feature]
        all_values = self.__input_dataset[self.__lag_short_rate_feature]

        # Plot where predictions are compared to all of the data
        plt.scatter(all_args, all_values, label='data')
        plt.plot(short_rate_grid, test_predictions, color='k', label='regression')
        plt.xlabel(self.__short_rate_feature)
        plt.ylabel(self.__lag_short_rate_feature)
        plt.title(f"{all}({self.skip_samples}, {self.take_samples})")
        plt.ylim([-2.5, 15])
        plt.legend()
        plt.show()
