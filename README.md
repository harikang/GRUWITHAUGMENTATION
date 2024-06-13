# GRUWITHAUGMENTATION

## Proposed Method

We apply a modified version of GRU with various Data Augmentation Methods. 

Firstly, our modification entails the integration of information from both the current \( t \) and preceding \( t-1 \) time steps. By updating the GRU Cell design to include information from two sequential time points, we aim to improve the learning of time sequences.

![modelflow5](https://github.com/harikang/GRUWITHAUGMENTATION/assets/138555992/29b8e9cf-b4b9-45d6-b1ff-80d8751c2ca9)

And We propose to use various data augmentation techniques to effectively train the feature extractor to learn more generalized and discriminative features and to enhance the model's robustness.

To start with, we add Gaussian noise to the feature data 
![gaussiannoise](https://github.com/harikang/GRUWITHAUGMENTATION/assets/138555992/272d7ffa-79a2-4532-9b1d-b0d95ca6318b)

and then apply the shifting technique that shifts the first 10 steps and the last 10 steps of this data forward or backward. 
![shifting](https://github.com/harikang/GRUWITHAUGMENTATION/assets/138555992/4a12d614-ab71-46ab-8516-9fb3abe127ea)

Lastly, we added Cutmix\cite{cutmix} which is traditionally known as an image augmentation technique.
![cutmix](https://github.com/harikang/GRUWITHAUGMENTATION/assets/138555992/9e253347-782d-4315-a8e4-403958543fd4)

In conclusion, The overall Data Augmentation method shown as below. 
![dataaugflow](https://github.com/harikang/GRUWITHAUGMENTATION/assets/138555992/acd61b70-3708-4aa8-bf89-0a8897ab93e3)

Our proposed method shows great improvement compared with State-of-the-Art(SOTA) with 96.76% accuracy.
![confusionmatrix](https://github.com/harikang/GRUWITHAUGMENTATION/assets/138555992/6af44e2d-f7b2-4817-bd31-6ef9ab4b2bbb)

