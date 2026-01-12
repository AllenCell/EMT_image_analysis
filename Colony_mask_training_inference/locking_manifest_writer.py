import argparse
import csv
import fcntl
import os


AICS_VAST_PREFIX = "/allen/aics"
MANIFEST_COLUMNS = ['File Path', 'VAST Path', 'File Name', 'Parent File Name']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvpath")
    parser.add_argument("--filepath", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--parentfilename", required=True)
    args = parser.parse_args()
    append_to_csv_manifest(args.csvpath, args.filepath, args.filename, args.parentfilename)


def append_to_csv_manifest(csv_manifest, file_path, file_name, parent_file_name):
    with open(csv_manifest, mode='a', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            size = os.fstat(f.fileno()).st_size
            writer = csv.writer(f)
            if size == 0:
                writer.writerow(MANIFEST_COLUMNS)
            writer.writerow([file_path_to_url(file_path), file_path, file_name, parent_file_name])
        except Exception as e:
            print(f"failed to write csv row for {csv_manifest} with {e}")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def file_path_to_url(file_path):
    if file_path.startswith(AICS_VAST_PREFIX):
        return file_path.replace(AICS_VAST_PREFIX, "https://vast-files.int.allencell.org")
    else:
        return file_path


if __name__ == '__main__':
    main()