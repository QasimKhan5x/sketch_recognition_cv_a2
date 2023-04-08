# Sketch Recognition
Sketch Recognition on TU Berlin Dataset with pre-trained ConvNets and Transfer Learning

## Usage

Download the weights for the model from this [link](https://drive.google.com/drive/folders/1KpsfKQNX8g5OwnIG8EOCBKnG-n6gUHuh?usp=sharing) and place the file in the root directory.

Install necessary requirements such as PyTorch, pandas, etc.

    pip install -r requirements.txt

Run `test.py`

    python test.py <src> <dst> <ckpt>

- For `src`, provide the path to the folder containing image files
- For `dst`, provide the name of the `.csv` file to be created
- For `ckpt`, provide the name of the downloaded `.ckpt` file