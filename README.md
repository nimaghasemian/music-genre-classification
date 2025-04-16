# Music Genre Classification with CNN


A deep learning application that classifies music into 10 different genres using Convolutional Neural Networks (CNN) and Mel-Frequency Cepstral Coefficients (MFCCs) features.

## Features

- 🎵 Classifies music into 10 genres: Blues, Classical, Country, Disco, Hip Hop, Jazz, Metal, Pop, Reggae, Rock
- 🎧 Supports both file upload and sample audio files
- 📊 Built with TensorFlow/Keras for the CNN model
- 🖥️ Streamlit-based user interface
- ⚡ Fast prediction with pre-trained model

## Prerequisites

- Python 3.8+
- pip package manager

## Installation

1. **Clone the Repository:**

   ```bash
    git clone https://github.com/nimaghasemian/Music-Genre-Classification.git
    cd Music-Genre-Classification
2. **Set Up Virtual Environment:**

   It is recommended to use a virtual environment:
   ```bash
   python -m venv music_venv
   source music_venv/bin/activate  # Windows: music_venv\Scripts\activate

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
4. **Run the Application:**
   ```bash
   streamlit run app.py


## Dataset
The dataset used in this project can be downloaded from the following link:
[GTZAN]((https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)))
- 1000 audio tracks (30 seconds each)
- 10 genres (100 tracks per genre)
- 22.05kHz sample rate
- 16-bit mono audio files

  
## Model Architecture  
  ```bash
  Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 13, 130, 32)      320       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 6, 65, 32)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 6, 65, 64)        18496     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 3, 32, 64)       0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 6144)              0         
                                                                 
 dense (Dense)               (None, 64)                393280    
                                                                 
 dropout (Dropout)           (None, 64)                0         
                                                                 
 dense_1 (Dense)             (None, 10)                650       
                                                                 
=================================================================
Total params: 412,746
Trainable params: 412,746
Non-trainable params: 0
```

## License
  This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).
