import argparse
import csv
import fcntl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvpath")
    parser.add_argument("csvcolumns", nargs="*")
    args = parser.parse_args()
    append_to_csv_manifest(args.csvpath, args.csvcolumns)


def append_to_csv_manifest(csv_manifest, column_entries):
    with open(csv_manifest, mode='a', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            writer = csv.writer(f)
            writer.writerow(column_entries)
        except:
            print(f"failed to write csv row for {csv_manifest}")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


if __name__ == '__main__':
    main()