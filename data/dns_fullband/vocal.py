import glob
import numpy as np


def run(path, seed):
    np.random.seed(seed)
    filenames = [f for f in glob.glob(path, recursive=True)]
    np.random.shuffle(filenames)
    test_data = int(0.01 * len(filenames))
    test_txt_data = filenames[:test_data]
    train_txt_data = filenames[test_data:]
    with open(r'vocal_test.txt', 'w') as fp:
        fp.write('\n'.join(test_txt_data))
    with open(r'vocal_train.txt', 'w') as fp:
        fp.write('\n'.join(train_txt_data))


if __name__ == "__main__":
    path = "vocal/*"
    seed = 99
    run(path, seed)
