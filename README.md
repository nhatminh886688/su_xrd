# spectralunmixing_xrd
This folder contains all the final versions of the code I wrote for the spectral unmxing project. There are two folders, the **data** folder contains all the data used for the code; the **code** folder contains all the codes. All the outputs of the codes are saved as .csv files in the **data** folder. 

There are 4 files in **code**, corresponding to the 4-state tissue classifcation process described in the paper. You should run the file in the following order: autoencoder.py, endmember_extraction.py, abundancefrac_estimation.py, classification.py.

As mentioned, the output of a Python script in the sequence would be saved as a csv file in data folder and this output is used as the input to the next script in the sequence until the final Accuracy is calculated in _classification.py_.

If you want to run the procedure on new data, place the new data in **data** folder and modify the input files appropriately in the above 4 scripts (follow the comments) and the code should work fine.

Dependencies: Tensorflow (1.12.0), Keras (2.2.4), Pandas, sklearn, nimfa, pysptools. Other than Tensorflow and Keras, all the remaining dependencies can be installed using the command ```pip install [dependency]```
