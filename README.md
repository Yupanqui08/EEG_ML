# EEG_ML
Dataset: https://zenodo.org/record/3554128#.Yn7MJi-cby8

Code originally from: https://www.kaggle.com/learn/intro-to-machine-learning

- DT_RFv1.py: Processes EEG data from subject 1 from the FEIS dataset using Decision Tree and Random Forest methods. Decision Tree method achieves about 69% accuracy and the Random Forest method acheives about 94% accuracy. Uses 14 channels as features. 
- output_v1: Output from DT_RFv1.py which shows accuracies. 


- DT_RFv3.ipynb: Features include all channels, hand (left, right, or ambidextrous), and language (English or Non-native speaker). Includes all subjects. Decision Tree accuracy: 0.5365  Random Forest Accuracy: 0.8607
