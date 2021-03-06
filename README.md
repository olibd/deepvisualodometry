# Deep Visual Odometry
This repository contains the code used during my master's research project.
The following package-class diagram gives an overview of the organization
of the project.

![DeepVisualOdometry Class diagram](https://user-images.githubusercontent.com/6964009/116123786-6fbc1d00-a691-11eb-9495-bb205ff6e661.png)

## SelfAttention
SelfAttentionVO, trained using an MSE-based loss, is the best performing model 
out of all the models tried during this research. It uses a multi-head attention 
module combined with an RCNN to estimate the visual odometry of a drone. The model 
takes 48% less time to trained compared to DeepVO and is 17% more accurate. It is 
also more robust to noisy inputs.

![SelfAttentionVO](https://user-images.githubusercontent.com/6964009/116123649-43a09c00-a691-11eb-9e03-a9e7abe55920.png)

Full details of the design and results can be found in the Thesis, which is available upon
request (will be made public later this spring).

## License
(c) Copyright @olibd, 2021
