# Data-Challenge 1 Group 15 Sprint 2

## Environment setup instructions
We recommend to set up a virtual Python 3.11 environment to install the package and its dependencies. To install the package, we recommend to execute `pip install -r requirements.txt` in the command line. This will install it in editable mode, meaning there is no need to reinstall after making changes. If you are using PyCharm, it should offer you the option to create a virtual environment from the requirements file on startup. Note that also in this case, it will still be necessary to run the pip command described above.

## Code structure
The code is structured into multiple files, based on their functionality. 
There are eleven `.py` files in total, each containing a different part of the code. The two `.ipynb` files contain some EDA of the dataset done by the group, but they are not mandatory to run.

- To download the data: 
  - Log in to Canvas, go to course JBG040 (2024-3) Data Challenge 1.
  - Go to the Files page. 
  - Download the four files `X_test.npy`, `X_train.npy`, `Y_test.npy` and `Y_train.npy`.
  - Create a folder called `data` in `dc1\`.
  - Move the files to the created folder `dc1\data\`.
  - You will only have to do this once usually, at the beginning of your project.


- To run everything in this project, follow these steps below in order. _Make sure the test and train data are downloaded beforehand!_
  **As the interpreter, select Python 3.11, as some packages are incompatible with newer Python versions.**
  - Open your terminal and execute `pip install -r requirements.txt` to install all dependencies/libraries needed for the project.
  - Run `image_augmentation.py` to convert the training dataset from multiple labels to binary labels (0 for No pneumothorax, 1 for Pneumothorax). After that, this code will do image augmentation to increase the number of images for the underrepresented class.
  - Run `binary_class.py` to run the whole training/evaluation pipeline.
  - Run `evaluate_model` to compute model evaluation metrics like Precision, Recall, Accuracy, F1-Score, Confusion matrix, AUROC curve, Precision-Recall curve.
  - Run `gradcam_visualization.py` to create Grad-CAM (XAI method) images. The created images are saved in `artifacts/gradcam`.
  - Run `lime_visualization.py` to create LIME (XAI method) images. The created images are saved in `artifacts/lime`.


