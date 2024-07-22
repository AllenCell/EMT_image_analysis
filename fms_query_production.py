# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: sm_fms_env
#     language: python
#     name: sm_fms_env
# ---

import aicsfiles
import pandas

FILE_DELIMETER = ","
ANNOTATION_VALUE_DELIMETER = ";"
SHOULD_WRITE_THIS_FILE = True
OUTPUT_PATH = "/allen/aics/assay-dev/users/Suraj/EMT_Work/BrightField/test_antoine_code_v2.csv"


def fms_file_to_readable_dict(file: aicsfiles.FMSFile) -> dict:
    annotations = {
        annotation: f'{ANNOTATION_VALUE_DELIMETER} '.join([f"{v}" for v in values])
        for annotation, values in file.annotations.items()
    }
    return {
        "id": file.id,
        "name": file.name,
        "path": file.path,
        "size": file.size,
        "type": file.type,
        "uploaded": file.uploaded,
        "thumbnail_path": file.thumbnail_path,
        **annotations,
    }


# +
# Retrieve good QC EMT files (including some from an undesired plate)
files_including_invalid_barcode = aicsfiles.fms.find(
    annotations={
        "Workflow": "EMT_deliverable_1",
        "Is Split Scene": True,
        "Presence of migration": True,
    }
)

# Filter out files with barcode "3500006060" from the query results
files = [
    fms_file_to_readable_dict(file)
    for file in files_including_invalid_barcode
    if "3500006060" not in file.get_annotation("Plate Barcode") and "3500005818" not in file.get_annotation("Plate Barcode")
]

df = pandas.DataFrame(files)

if SHOULD_WRITE_THIS_FILE:
    df.to_csv(OUTPUT_PATH, sep=FILE_DELIMETER)
# -


