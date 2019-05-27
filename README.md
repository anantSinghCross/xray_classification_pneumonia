# Classification between Normal and Pneumoniac X-rays

The main goal is to classify an X-Ray to tell whether it's normal or pneumoniac. Currently the error percentage is as high as **34%**. That is because I had to compress the original dataset (available at *Kaggle* https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/downloads/chest-xray-pneumonia.zip/2#chest_xray.zip) since it was the only way to train the model on my machine in a feasible time.

### Dataset

The original dataset has images upto *2000 by 2000* in size so I resized the images to *60 by 60* and trained the model using them. Now resizing them to this size means there will be a huge loss of information. which is the main reason for this model to perform poorly.

#### Note

Currently the model uses only one CNN layer. I'll try to improve the model as I learn more about using multiple CNN layers. Plus I'll try not to compromise the dataset to this extent lol :D

This was only an attempt to see if I have actually learnt something from my past projects.
