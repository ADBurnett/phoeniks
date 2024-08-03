import re
import csv
import numpy as np
from .thz_data import Data
from scipy import interpolate




def determine_header_lines(file_path):
    """
    Reads a file and counts the number of lines that don't contain only digits, '.', '-', or 'e'.

    Args:
        file_path (str): Path to the input file.

    Returns:
        int: Number of lines that don't meet the criteria.
    """
    try:
        with open(file_path, 'r') as file:
            for line_number, line in enumerate(file, start=0):
                # Remove leading/trailing whitespace and check if the line meets the criteria
                cleaned_line = re.sub(r"[\n\t\s.\-eE]*", "", line)
                if cleaned_line.isdigit():                    
                    return line_number 
        return 0  # All lines meet the criteria
    except FileNotFoundError:
        return None  # File not found
    
def find_delimiter(filename):
    sniffer = csv.Sniffer()
    try:
        with open(filename) as fp:
            delimiter = sniffer.sniff(fp.read(5000)).delimiter
            return delimiter
    except FileNotFoundError:
        return None
    

import numpy as np

def read_Leeds(filename):
    delimiter = find_delimiter(filename)
    headers = determine_header_lines(filename)
    array = np.genfromtxt(filename, dtype='float', comments='#', delimiter=delimiter, skip_header=headers, usecols=(0,1,2))
    time, data, std = array.T
    time = time * 1E-12

    return time, data, std



def read_XY(filename):
    delimiter = find_delimiter(filename)
    headers = determine_header_lines(filename)
    array = np.genfromtxt(filename, dtype='float', comments='#', delimiter=delimiter, skip_header=headers, usecols=(0,1))
    time, data = array.T
    time = time * 1E-12
    std = None

    return time, data, std


def read_XYYY(filename):
    delimiter = find_delimiter(filename)
    headers = determine_header_lines(filename)
    array = np.genfromtxt(filename, dtype='float', comments='#', delimiter=delimiter, skip_header=headers)
    array = array.T
    time = array[:,[0]]
    data = np.mean(array[:,1:])
    std = np.std(array[:,1:])
    time = time * 1E-12

    return time, data, std


def read_fd_XY(filename):
    delimiter = find_delimiter(filename)
    headers = determine_header_lines(filename)
    real, imag = np.genfromtxt(filename, dtype='float', comments='#', delimiter=delimiter, skip_header=headers)
    data = np.empty(real.shape, dtype=complex)
    data.real = real
    data.imag = imag

    return data


def read_XYXY(filename, XYXYargs=[0.1,100,0.001]):
    #delimiter = find_delimiter(filename)
    delimiter ="\t"
    headers = determine_header_lines(filename)
    array = np.genfromtxt(filename, dtype='float', comments='#', delimiter=delimiter, skip_header=headers)
    #array = array.T
    x_array = array[:, 0::2]  # Select every second column (x values)
    y_array = array[:, 1::2]  # Select every second column (y values)

    #check if all columns in X are the same

    are_columns_identical = np.all(x_array[:, 1:] == x_array[:, :-1])

    if are_columns_identical:
        time = x_array[:, 0]
        data = np.mean(y_array)
        std = np.std(y_array)
    else:

        x_start = XYXYargs[0]
        x_stop = XYXYargs[1]
        x_spacing = XYXYargs[2]

        # Create the new x-axis values
        new_time = np.arange(x_start, x_stop, x_spacing)

        interpolated_y_values = np.zeros((y_array.shape[1], len(new_time)))

        # Interpolate y-values onto the new x-axis
        for col in range(y_array.shape[1]):
            interp_func = interpolate.interp1d(x_array[:, col], y_array[:, col], kind='linear')
            interpolated_y_values[col, :] = interp_func(new_time)

        interpolated_y_values = interpolated_y_values.T
        time = new_time
        data = np.mean(interpolated_y_values)
        std = np.std(interpolated_y_values)


    return time, data, std


def read_VertFile(filename):
    delimiter ="\t"
    headers = 2
    filename = "Data/new_data/VertSweep_320V_10^7VpA_purged_tc5ms_Gathering_Quartz_ref.dat"
    array = np.genfromtxt(filename, dtype='float', comments='#', delimiter=delimiter, skip_header=headers)
    time = array[1:, 0]
    time = time * 1E-12
    y_array = array[:, 1::2]  # Select every second column (y values)

    unique_values = np.unique(y_array[0])

    # Create a dictionary to hold the split arrays
    data = {}

    # Create a dictionary to hold the mean and standard deviation
    stats = {}

    # Iterate over unique values and create arrays
    for value in unique_values:
        # Select columns where the top row is equal to the current unique value
        data[value] = y_array[1:, y_array[0] == value]


        # Calculate mean and standard deviation for each array
        mean = np.mean(data[value], axis=1)
        std_dev = np.std(data[value], axis=1)
        
        # Store the mean and standard deviation in the stats dictionary
        stats[value] = {'mean': mean, 'std_dev': std_dev}

    return time, data, stats, unique_values



def create_data(ref_file, sample_file, dark_file=None, fd_reference_std=None, fd_sample_std=None, fd_dark_std=None, reader='Leeds', sample_thickness=None, sample_name=None, XYXYargs=[None,None,None]):
    
    reader_functions = {
    'Leeds': read_Leeds,
    'XY': read_XY,
    'XYYY': read_XYYY,
    'XYXY' : read_XYXY,
    # Add other window types here

}
    
    reader_func = reader_functions.get(reader)

    if reader_func == read_XYXY:
        time_ref, data_ref, std_ref = reader_func(ref_file, XYXYargs=[None,None,None])
        time_samp, data_samp, std_samp = reader_func(sample_file, XYXYargs=[None,None,None])
        if dark_file != None:
            time_dark, data_dark, std_dark = reader_func(dark_file, XYXYargs=[None,None,None])
        else:
            time_dark, data_dark, std_dark = None, None, None  
    else:
        time_ref, data_ref, std_ref = reader_func(ref_file)
        time_samp, data_samp, std_samp = reader_func(sample_file)

        if dark_file != None:
            time_dark, data_dark, std_dark = reader_func(dark_file)
        else:
            time_dark, data_dark, std_dark = None, None, None

    if np.array_equal(time_ref, time_samp):
        if dark_file != None:
            if np.array_equal(time_ref, time_dark) and np.array_equal(time_samp, time_dark):
                data = Data(time = time_ref, td_reference = data_ref, td_sample = data_samp, td_dark= data_dark, td_ref_std=std_ref, td_samp_std=std_samp, td_dark_std=std_dark, thickness = sample_thickness)
            else:
                print("Time in referance/sample and dark measurement are not the same")
        else:
            data = Data(time = time_ref, td_reference = data_ref, td_sample = data_samp, td_ref_std=std_ref, td_samp_std=std_samp, thickness = sample_thickness)
        if fd_reference_std is None or fd_sample_std is None or fd_dark_std is None:
            data.fd_reference_std = None
            data.fd_sample_std = None
            data.fd_dark_std = None
            data.fd_reference_std_raw = data.fd_reference_std
            data.fd_sample_std_raw = data.fd_sample_std
            data.fd_dark_std_raw = data.fd_dark_std
            data.mode = "reference_sample_dark"
        else:
            data.fd_reference_std = read_fd_XY(fd_reference_std)
            data.fd_sample_std = read_fd_XY(fd_sample_std)
            data.fd_dark_std = read_fd_XY(fd_dark_std)
            data.fd_reference_std_raw = data.fd_reference_std
            data.fd_sample_std_raw = data.fd_sample_std
            data.fd_dark_std_raw = data.fd_dark_std
            data.mode = "reference_sample_dark_standard_deviations"

    else:
        print("Time in referance and sample measurement are not the same")

    

    return data