def run():
    filenames = ['vctk/train.txt', 'dns_fullband/emotional_train.txt',
                 'dns_fullband/french_train.txt', 'dns_fullband/read_speech_train.txt',
                 'dns_fullband/russian_train.txt', 'dns_fullband/vocal_train.txt']
    with open('large_fullband_train.txt', 'w') as outfile:
        for name in filenames:
            with open(name) as infile:
                for line in infile:
                    if ".wav" in line:
                        outfile.write(line)
                outfile.write("\n")


if __name__ == "__main__":
    run()
