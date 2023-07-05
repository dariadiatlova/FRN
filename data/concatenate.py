def run():
    filenames = ['vctk/test.txt', 'dns_fullband/emotional_test.txt',
                 'dns_fullband/french_test.txt', 'dns_fullband/read_speech_test.txt',
                 'dns_fullband/russian_test.txt', 'dns_fullband/vocal_test.txt']
    with open('large_fullband_test.txt', 'w') as outfile:
        for name in filenames:
            with open(name) as infile:
                for line in infile:
                    if ".wav" in line:
                        outfile.write(line)
                outfile.write("\n")

    with open('large_fullband_test.txt', 'r') as outfile:
        for i, line in enumerate(outfile):
            if ".wav" not in line:
                print(f"Line {i} is {line}")


if __name__ == "__main__":
    run()
