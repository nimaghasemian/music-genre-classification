import numpy as np 
import librosa
import math
from tensorflow import keras
import streamlit as st
import time
import pyautogui

def get_mfcc(audio_signal, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """
    Extract MFCCs from an audio file.
    """
    new_data = {"mfcc": []}
    SAMPLE_RATE = 22050
    # Load audio file using the specified sample rate
    signal, sample_rate = librosa.load(audio_signal, sr=SAMPLE_RATE)
    TRACK_DURATION = int(librosa.get_duration(y=signal, sr=sample_rate))  # measured in seconds
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    for d in range(num_segments):
        start = samples_per_segment * d
        finish = start + samples_per_segment
        # Extract MFCCs using keyword arguments for y and sr
        mfcc = librosa.feature.mfcc(
            y=signal[start:finish], 
            sr=sample_rate, 
            n_mfcc=num_mfcc, 
            n_fft=n_fft, 
            hop_length=hop_length
        )
        mfcc = mfcc.T
        # Store only if the number of vectors matches the expected count
        if len(mfcc) == num_mfcc_vectors_per_segment:
            new_data["mfcc"].append(mfcc.tolist())
    return new_data["mfcc"]

def prediction(mfcc):
    """
    Predict the genre using the pre-trained CNN model.
    """
    # Use the H5 file since Keras 3 supports legacy H5 files.
    cnn_model = keras.models.load_model('music-gen-classify-v2.h5')
    mfcc = np.array(mfcc)
    mfcc = mfcc[..., np.newaxis]
    pred = cnn_model.predict(mfcc)
    return max(np.argmax(pred, axis=1))

def get_genre(prediction):
    """
    Convert prediction index to genre name.
    """
    if prediction == 0:
        return 'Blues'
    elif prediction == 1:
        return 'Classical'
    elif prediction == 2:
        return 'Country'
    elif prediction == 3:
        return 'Disco'
    elif prediction == 4:
        return 'Hip Hop'
    elif prediction == 5:
        return 'Jazz'
    elif prediction == 6:
        return 'Metal'
    elif prediction == 7:
        return 'Pop'
    elif prediction == 8:
        return 'Reggae'
    elif prediction == 9:
        return 'Rock'
    else:
        return 'Unknown'

def main():
    st.set_page_config(layout='wide', page_title='Genre Classification', page_icon='🎵')
    st.title('Music Genre Classification With CNN')
    st.markdown(
        'We use **GTZAN** Dataset which is a very popular dataset for Audio Classification. '
        'The uploaded audio file should be less than **30sec** and in **.WAV** format for best results. '
        'For best performance, provide an instrumental segment. '
        'If you want to test the model, select ***Untrained Samples***. '
        'The model currently supports 10 genres: Blues, Classical, Country, Disco, Hip Hop, Jazz, Metal, Pop, Reggae, and Rock. '
        'A work by Mir Nima Ghasemian'
        
    )
    
    selected_item = st.selectbox('Select Either Uploaded Samples or Upload', ['Untrained Samples', 'Upload'])
    
    if selected_item is not None:
        if selected_item == 'Upload':
            files = st.file_uploader('Select .WAV File with maximum 30sec Time', type='wav', accept_multiple_files=False)
            if files is not None:
                # Load audio and determine its duration using keyword arguments
                audio, sr = librosa.load(files, sr=22050)
                duration = int(librosa.get_duration(y=audio, sr=sr))
                if 'file_uploaded' not in st.session_state:
                    st.session_state['file_uploaded'] = True
                if duration > 30:
                    st.session_state['file_uploaded'] = False
                    st.write('Reupload file as it exceeds the time limit.')
                    bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        bar.progress(i + 1)
                    pyautogui.hotkey('ctrl', 'F5')
                elif st.session_state.get('file_uploaded', False):
                    st.audio(files, format="audio/wav", start_time=0)
        elif selected_item == 'Untrained Samples':
            selected_file = st.selectbox("Select A Sample", ['Blues', 'Jazz', 'Country', 'Classical', 'Hiphop', 'Metal', 'Pop', 'Reggae', 'Rock'])
            files = f'Data/upload/user/{selected_file}.wav'
            st.audio(files, format="audio/wav", start_time=0)
        submitted = st.button("Submit")
    
    if submitted:
        with st.spinner('Model is trying to predict your genre! Please wait...'):
            signal = files
            mfcc_for_track = get_mfcc(signal)
            pred = prediction(mfcc_for_track)
            genre = get_genre(int(pred))
        st.success('Prediction complete!')
        st.markdown(f'The genre for your music is 🎵: **{genre}** Music')

if __name__ == '__main__':
    main()
