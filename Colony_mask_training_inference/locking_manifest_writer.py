import argparse
import csv
import fcntl
import os


MANIFEST_COLUMNS = ['File Path', 'File Name', 'Parent File Name']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvpath")
    parser.add_argument("--filepath", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--parentfilename", required=True)
    args = parser.parse_args()
    append_to_csv_manifest(args.csvpath, args.csvcolumns)


def append_to_csv_manifest(csv_manifest, file_path, file_name, parent_file_name):
    with open(csv_manifest, mode='a', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            size = os.fstat(f.fileno()).st_size
            writer = csv.writer(f)
            if size == 0:
                writer.writerow(MANIFEST_COLUMNS)
            writer.writerow([file_path, file_name, parent_file_name])
        except:
            print(f"failed to write csv row for {csv_manifest}")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


if __name__ == '__main__':
    main()