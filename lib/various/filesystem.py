import os


def find_calibration_folder(recording_folder):
    """
    Find the calibration subfolder within the recording folder.

    Args:
        recording_folder (str): Path to the recording folder

    Returns:
        str: Name of the calibration subfolder (not the full path)

    Raises:
        FileNotFoundError: If no calibration folder is found
    """
    calibfolder = ""

    for x in os.listdir(recording_folder):
        if "calibration" in x:
            calibfolder = x

    if calibfolder == "":
        raise FileNotFoundError("Could not find calibration folder in {0}".format(recording_folder))

    return calibfolder
