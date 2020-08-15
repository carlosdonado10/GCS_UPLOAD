import os

from utils.preprocessing_utils import extract_images, sample_image
import pickle
import numpy as np

def main():
    checkpoint = np.array([])
    if 'checkpoint.npy' in os.listdir('utils'):
        checkpoint = np.load(os.path.join('utils', 'checkpoint.npy'))
    try:
        folds = [dirname for dirname in os.listdir() if dirname.startswith('Fold')]

        for fold in folds:
            participants = [dirname for dirname in os.listdir(fold) if len(dirname) == 2]

            for participant in participants:
                videos = [dirname for dirname in os.listdir(os.path.join(fold, participant)) if
                          dirname.endswith('mp4') or dirname.endswith('MOV') or dirname.endswith('mov') or dirname.endswith('MP4')]

                for video in videos:
                    if os.path.join(fold, participant, video) not in checkpoint:
                        label = int(video.split('.')[0]) // 5
                        print(f'Uploading frames from {os.path.join(fold, participant, video)}... :)')
                        extract_images(pathIn=os.path.join(fold, participant, video), pathOut='./processed_images', fps=1,
                                       label=label)
                        checkpoint = np.append(checkpoint, os.path.join(fold, participant, video))
                        np.save(os.path.join('utils', 'checkpoint'), checkpoint)
                    else:
                        print(f'skipping {os.path.join(fold, participant, video)} -> Already uploaded frames :D')

    except Exception as e:
        raise e
    finally:
        np.save(os.path.join('utils', 'checkpoint'), checkpoint)


def define_rotations():

    folds = [dirname for dirname in os.listdir() if dirname.startswith('Fold')]

    for fold in folds:
        participants = [dirname for dirname in os.listdir(fold) if len(dirname) == 2]
        for participant in participants:
            videos = [dirname for dirname in os.listdir(os.path.join(fold, participant)) if dirname.endswith('mp4') or dirname.endswith('MOV') or dirname.endswith('mov') or dirname.endswith('MP4')]
            for video in videos:
                sample_image(pathIn=os.path.join(fold, participant, video))


if __name__ == '__main__':

    #define_rotations()
    main()

