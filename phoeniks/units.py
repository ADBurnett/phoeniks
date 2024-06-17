possible_frequency_units = ['wavenumber','THz','GHz','ang','nm','um','mm','cm','m']
possible_length_units = []
possible_time_units = []

def isThisAFrequencyUnit(unit):
    '''
    Return true if this is a frequency unit, false if a wavelength.

    Units of frequency are 'wavenumber','THz','GHz'
    Units of wavelength are 'ang','nm','um','mm','cm' or 'm'

    Returns
    -------
    bool
        True if this is a frequency unit, False if a wavelength.
    '''
    index = possible_frequency_units.index(unit)
    if index <=2:
        result = True
    else:
        result = False
    return result

def convert_frequency_units( value, unit_in, unit_out ):
    '''
    Convert between frequency and wavelength units.

    The input can be either a scalar value or a numpy array of values. The function will return the converted value(s) in the output units specified.
    The unit strings are turned into lower-case so case is irrelevant

    Parameters
    ----------
    unit_in : str
        The units of the input value(s). Can be one of 'cm-1' (or 'wavenumber'), 'GHz', 'THz', 'nm', 'um', 'mm', 'cm', 'm'.
    unit_out : str
        The units of the output value(s). Must be one of 'cm-1' (or 'wavenumber'), 'GHz', 'THz', 'nm', 'um', 'mm', 'cm', 'm'.
    input_value : scalar or numpy array
        The value(s) for which the conversion is to be made.

    Returns
    -------
    scalar or numpy array
        The converted value(s) in the output units specified.

    '''

    unit_in  = unit_in.lower()
    unit_out = unit_out.lower()
    # the conversion dictionary has a tuple for each unit
    # the first entry is the multiplicative factor, the second is the operator
    # To convert wavelength they have to be scaled and then inverted
    wavenumber = { 'cm-1'       : (1,'*'),
                   'wavenumber' : (1,'*'),
                   'thz'        : (33.356,'*'),
                   'ghz'        : (33.356E+3,'*'),
                   'ang'        : (1E-8,'/'),
                   'nm'         : (1E-7,'/'),
                   'um'         : (1E-4,'/'),
                   'mm'         : (1E-1,'/'),
                   'cm'         : (1.0 ,'/'),
                   'm'          : (1E2 ,'/'),
                 }
    # Convert the input unit to cm-1
    if isinstance(value,np.ndarray):
        for i,x in enumerate(value):
            if x <= 0:
                # Problem with negative or zero conversions between wavelength and frequency
                print('Warning a zero or negative frequency/wavelength is not permitted',i,x,unit_in,unit_out)
                x[i] = 1.0E-8
    else:
        if value <= 0 :
            value = 1.0E-8
    scale,operator = wavenumber[unit_in]
    value = scale * value     
    if operator == '/':
        value = 1.0 / value  
    # convert the internal value from cm-1 to the output unit
    scale,operator = wavenumber[unit_out]
    if operator == '/':
        value = 1.0 / value  
    value = value / scale
    return value