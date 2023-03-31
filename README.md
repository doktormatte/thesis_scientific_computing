# Deep Learning for Electric Vehicle Energy Demand and Charging Station Occupancy Forecasting

Framework for performing data pre-processing, hyperparameter search and forecasting evaluation of deep learning models as described within the thesis.

The original event-based data is stored in the following files:
- acn_caltech.csv
- acn_jpl.csv
- boulder.csv
- palo_alto.csv

"data_preprocessing.py" generates training and test datasets from these files, and defines the exported file names and directory structures.

Hyperparameter search is performed by "DL_hypersearch.py", which utilizes the generated training datasets and create a hold-out validation set. 

"DL_forecasting.py" selects model configurations with the lowest validation loss, trains them on the full training datasets, and evaluates their forecasting performance at three different horizons.

Results of hyperparameter search and evaluation are read from and written to four central CSV files. Access to these files is managed by functions within each script that check and wait for file availability:
- acn_caltech_real.csv
- acn_jpl_real.csv
- boulder_real.csv
- palo_alto_real.csv 

Trained models are exported to a "models" directory.

The framework supports parallel execution on multiple machines or environments without redundant computations, and scripts terminate automatically once experiments are completed. Hyperparameter search should be completed before starting the forecasting evaluation. 

Memory usage is reduced through Keras data generators and the TensorFlow Mixed Precision API.

Additional features not mentioned in the thesis include the CNN-GRU and CNN-BiLSTM models, and the ability to train models in "stateful" mode using Keras. For stateful mode to function, a fixed batch size is needed, which was set to 64 during the thesis experiments. In this mode, training and test datasets must be set to a fixed size that is evenly divisible by the selected batch size.

All implemented models function in both stateless and stateful modes.

The conda environments for both gpu- and cpu-based learning in TensorFlow are included. Tensorflow-gpu was installed using the following instructions: https://www.tensorflow.org/install/pip.
