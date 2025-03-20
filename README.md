# Data-Challenge 1 Group 15 Sprint 1

## Environment setup instructions
We recommend to set up a virtual Python 3.11 environment to install the package and its dependencies. To install the package, we recommend to execute `pip install -r requirements.txt` in the command line. This will install it in editable mode, meaning there is no need to reinstall after making changes. If you are using PyCharm, it should offer you the option to create a virtual environment from the requirements file on startup. Note that also in this case, it will still be necessary to run the pip command described above.

## Code structure
The code is structured into multiple files, based on their functionality. 
There are five `.py` files in total, each containing a different part of the code. 
Feel free to create new files to explore the data or experiment with other ideas.

- To download the data: 
  - Log in to Canvas, go to course JBG040 (2024-3) Data Challenge 1.
  - Go to the Files page. 
  - Download the four files `X_test.npy`, `X_train.npy`, `Y_test.npy` and `Y_train.npy`.
  - Create a folder called `data` in `dc1\`.
  - Move the files to the created folder `dc1\data\`.
  - You will only have to do this once usually, at the beginning of your project.

- To run the whole training/evaluation pipeline: run `main.py`. **As the interpreter, select Python 3.11, as some packages are incompatible with newer Python versions.**
This script does the following things:
    - It loads your train and test data. _Make sure its downloaded beforehand!_
    - It initializes the neural network as defined in the `Net.py` file.
    - It initializes the loss functions and optimizers. If you want to change the loss function/optimizer, do it here.
    - It defines the number of training epochs and the batch size.
    - It checks and enables GPU acceleration for training (if you have a CUDA or Apple Silicon enabled device).
    - It trains the neural network and performs evaluation on the test set at the end of each epoch.
    - It provides plots about the training losses both during training in the command line and as a png, saved in the `\artifacts\` subdirectory.
    - Finally, it saves your trained model's weights in the `\model_weights\` subdirectory so that you can reload them later.
