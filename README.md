# Install Dependencies
conda create -n saturated_utility python==3.7
conda activate saturated_utility

## has gpu, and older than 3080/3090

conda install cudatoolkit=10.0
conda install cudnn
pip install tensorflow-gpu==1.15

conda install matplotlib
conda install requests
conda install jupyter


## Apple M1
Follow the instructions at

https://developer.apple.com/metal/tensorflow-plugin/

conda install scikit-learn, matplotlib, pandas