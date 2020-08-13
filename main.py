import os

from utils.preprocessing_utils import extract_images, sample_image


def main():
    folds = [dirname for dirname in os.listdir() if dirname.startswith('Fold')]

    for fold in folds:
        participants = [dirname for dirname in os.listdir(fold) if len(dirname) == 2]

        for participant in participants:
            videos = [dirname for dirname in os.listdir(os.path.join(fold, participant)) if
                      dirname.endswith('mp4') or dirname.endswith('MOV') or dirname.endswith('mov') or dirname.endswith('MP4')]
            for video in videos:
                label = int(video[0]) // 5
                extract_images(pathIn=os.path.join(fold, participant, video), pathOut='./processed_images', fps=1,
                               label=label)


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

