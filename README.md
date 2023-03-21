
# Sugar Prediction Model

This repository holds the code used for a sugar position prediction model. This work is part of a PhD project funded by the BBSRC.

## Requirements
 - [Tensorflow](https://github.com/tensorflow/tensorflow) - with GPU installation
 - [tensorflow_addons](https://github.com/tensorflow/addons)
	 -  `pip install tensorflow_addons`
 - [Gemmi](https://github.com/project-gemmi/gemmi) 
	 -  `pip install gemmi`
 - [tqdm](https://github.com/tqdm/tqdm)
	  -  `pip install tqdm`
 
## Usage
For any code to run with model, activate tensorflow conda environment: 

	conda activate tf

To train model run:

    python scripts/interpolated_model/train_interpolated.py

To predict using model run: 

	python scripts/interpolated_model/predict_interpolated.py

To evaluate prediction: 

	python scripts/interpolated_model/test_model.py
