# AIDA-UPM-SemEval-2022-Task-5-MAMI-
AIDA-UPM at SemEval-2022 Task 5: Exploring Multimodal Late Information Fusion for Multimedia Automatic Misogyny Identification


## Description

This repository holds the research outcomes obtained during the participation at the SemEval-2022 Multimedia Automatic Misogyny Identification (MAMI) task. 

Our main contribution is the exploration of different late fusion methods to boost the performance of the combination based on the Transformer-based model and Convolutional Neural Networks (CNN) for text and image, respectively. Additionally, our findings contribute to better understanding the effects of different image preprocessing methods for meme classification. We achieve $0.636$ F1-macro average score for the binary subtask A, and $0.632$ F1-macro average score for the multi-label subtask B. The present findings might help solve the inequality and discrimination women suffer on social media platforms.

Here we show a summarized view of our proposed late fusion approach:

![image](https://user-images.githubusercontent.com/56938752/153618208-be62c7db-f28c-450c-9dd9-737a954fb221.png)


## Resources

## Inpainting Pre-processing
We carried out three different pre-processing steps for image classification: no preprocessing, (2) blacking out, and (3) inpainting the text from the image.
You can explore this last method in our Hugging Face Space: https://huggingface.co/spaces/Huertas97/Inpaint_Me

## Auto-sklearn Late Fusion models

The Auto-sklearn package is used in the multimodal late fusion step.  It automatically explore a wide range of models and preprocessing approaches available in scikit-learn and identify the best ensemble configuration. We opted for this method because it implements Bayesian Optimization for searching the optimal pipeline configuration and Ensemble Selection to choose the suitable model.

You can load the models from pickle files as: 

````python
import pickle

path = "./Models/clf-late-fusion-A.pkl"
with open(path, "rb") as f:
    clf = pickle.load(f)
````

Moroever, you can inspect the ML models that compose the best ensemble as shown below. You will need `autosklearn >= 0.14.3`

````python
import autosklearn

print(clf.show_models())
````

