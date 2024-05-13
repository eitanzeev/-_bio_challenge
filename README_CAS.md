Hi! Welcome to my ReadME.md. 

This will serve as a brief description of my work. 

I split my code into 4 files. 
1. `a_alpha_data_exploration.ipynb`
    - This notebook is your main stop for understanding my work. 
    - Run through the notebook sequentially to re-create my work. Alternatively, run specific experiments by running all code up to the "Modelling" Section divide. Then Run the parameters intialization.
    - The notebook is divided into
        *  Package Imports
        *  Data Import and Exaination 
            *  Handling missing Kd measurements
            *  Duplicate detection
            *  Distribution Analyses
            *  Feature generation
            *  Correlation Studies
            
        * Data Selection
            *  q_value filtering 
            *  Amino acid frequency scaling
            *  Embedding indices vs One Hot encoding 
            *  Training Slices
            *  Prepare each 
        *  Modelling
            *  DeepLearning Approaches  
                *  CNN Experiments (I did not do hyperparameter tuning of DL approach given time and compute restraints but mention parameters to tune. )
                *  RNN Experiments 
                *  Classic ML Approaches
        *  HoldOut Work
            *  Me realizing that the sequences are not uniform like in the training set. I got got.
            *  Model re-training with padding + evaluation
            *  HoldOut set Prediction

2. `modelling_utilities.py` - Supporting functions for data loader construction, training loop execution, model instantiation, etc. 
3. `model_architectures.py` - Class definitions for my CNN and RNN definitions. Classic ML models are smaller and can be defined and run from a_alpha_data_exploration.ipynb
    *  This script may be helpful to get a deeper understnading of how my deep learning models were created. 
4. `plotting_assistance.py` - script for creating graphs.
5. `results_exploration.ipynb` - A notebook summarizing the results for each experiment for easier reporting in ppt 



Models are stored in `./models`. Model results are stored in `./model_results`. 

Models are named dynamically per experiment based on user entries! I adjust model saving to save per fold but that can adjusted. 


I spent quite a bit of time on making my architecures dynamic and easily adjustable to user preferences. My goal was to demonstrate my understanding of deep learning architectures while highlighing my experience as a Machine Learning Engineer with experience in pipeline development. My opinion is that often times deep learning knowledge and expertise are hamstrung by bad engineering. I tried to demosntrate that I can bridge that gap!
