import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import argparse
import pandas as pd


def get_features_from_wav(wav_path, sec):
    """
    Samples audio by given time window

    :param wav_path: path to .wav file
    :param sec: float, sampling frame size in sec
    :return: pandas.DataFrame with sampled audio of shape (n_samples, frames_per_sample)
    """
    audio, sr  = librosa.load(wav_path)
    # short_frame = rate * sec
    S = librosa.feature.melspectrogram(audio, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    delta_mfcc  = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    res_X = np.vstack((mfcc, delta_mfcc, delta2_mfcc)).T
    res_X = StandardScaler().fit_transform(res_X)
    res_df = pd.DataFrame(res_X)
    colnames = ["mfcc_{}".format(i) for i in range(mfcc.shape[0])] +\
               ["mfcc_delta_{}".format(i) for i in range(delta_mfcc.shape[0])]  +\
               ["mfcc_delta2_{}".format(i) for i in range(delta2_mfcc.shape[0])] 
    res_df.columns = colnames
    return res_df


def main():
    parser = argparse.ArgumentParser(description='Feature extraction script based on PythonAudioAnalysis features')
    parser.add_argument('--frame_ms', type=int, default=10,
                        help='Length of each frame in ms')
    parser.add_argument('--wav_path', type=str, help='Path to .wav dile')
    parser.add_argument('--feature_save_path', type=str, help='Path to save features .csv file')
    args = parser.parse_args()

    feature_df = get_features_from_wav(args.wav_path, 0.001 * args.frame_ms)
    print("Created features dataframe with shape:", feature_df.shape)

    print("Saving features:", args.feature_save_path)
    feature_df.to_csv(args.feature_save_path, index=False)
    print("Done")

if __name__ == '__main__':
    main()
