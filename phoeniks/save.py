import dill
import pandas as pd
from dash import Dash
import dash_bootstrap_components as dbc
import phoeniks as pk


def save_data_to_pickle(data_class, filename):
    """
    Saves an instance of a data class to a pickle file using dill.

    Args:
        data_class_name (str): Name of the data class (e.g., "MyDataClass").
        filename (str): Name of the pickle file to save.

    Returns:
        None
    """
    #try:
    # Dynamically import the data class

    if isinstance(data_class, pk.thz_data.Data):
    
        if hasattr(data_class, 'app'):
            data_class.app = 'dill removed this attribute'
        else:
            pass
        
        # Create an instance of the data class (you can customize this part)
        #instance = data_class()

        # Save the instance to the pickle file
        with open(filename, 'wb') as file:
            dill.dump(data_class, file)

        print(f"Instance of {data_class} saved to {filename} using dill.")
    else:
         print("Error: data_class should be a class and is a", type(data_class))



def read_pickle_to_data(filename):
    """
    reads an instance of a pickle file and saves a data class using dill.

    Args:
        data_class_name (str): Name of the data class (e.g., "MyDataClass").
        filename (str): Name of the pickle file to save.

    Returns:
        None
    """
    try:
        # Save the instance to the pickle file
        with open(filename, 'rb') as file:
            loaded_data = dill.load(file)

        
        print(f"Instance of {filename} read in using dill.")
    except KeyError:
        print(f"Error: Data class '{filename}' not found.")


    external_stylesheets = [dbc.themes.BOOTSTRAP]
        
    loaded_data.app = Dash(__name__, external_stylesheets=external_stylesheets)

    return loaded_data

def save_to_file(data, filename, Type ='frequency', Smooth=False):

    if Type == 'frequency':

        df_all_spec = {"frequency" : data.frequency*1E-12, "n" : data.n, "k" : data.k, "Absorption Coefficient" : data.a/100}
        df_all_spec = pd.DataFrame(df_all_spec)
        df_all_spec.columns = pd.MultiIndex.from_tuples([("Frequency", 'THz'), ('Refractive Index', 'dimensionless'), ('Extinction Coefficient', 'dimensionless'),('Absorption Coefficient', 'cm^-1')])
        df_all_spec.set_index(df_all_spec.columns[0], drop=False, inplace=True)

        temp = []
        df_all_smooth = pd.DataFrame(temp)

        if hasattr(data, 'ecmplx'):
                data_perm = {"frequency" : data.frequency*1E-12, "e_real" : data.ereal, "e_imag" : data.eimag}
                df_perm = pd.DataFrame(data_perm)
                df_perm.columns = pd.MultiIndex.from_tuples([("Frequency", 'THz'), ('Real Permittivity', 'dimensionless'), ('Imaginary Permittivity', 'dimensionless')]) 
                df_perm.set_index(df_perm.columns[0], drop=True, inplace=True) 
                df_all_spec = pd.concat([df_all_spec, df_perm], axis=1)
        else:
                pass

        if hasattr(data, 'alpha_smooth'):
                data_smooth = {"frequency" : data.frequency*1E-12, "n_smooth" : data.n_smooth, "k_smooth" : data.k_smooth, "a_smooth" : data.alpha_smooth}
                df_smooth = pd.DataFrame(data_smooth)
                df_smooth.columns = pd.MultiIndex.from_tuples([("Frequency", 'THz'), ('Smoothed Refractive Index', 'dimensionless'), ('Smoothed Extinction Coefficient', 'dimensionless'),('Smoothed Absorption Coefficient', 'cm^-1')])  
                df_smooth.set_index(df_perm.columns[0], drop=False, inplace=True)
                df_all_smooth = df_smooth
        else:
                pass

        if hasattr(data, 'ecmplx_smooth'):
                data_cmplx_smooth = {"frequency" : data.frequency*1E-12, "e_real" : data.ereal_smooth, "e_imag" : data.eimag_smooth}
                df_cmplx_smooth = pd.DataFrame(data_cmplx_smooth)
                df_cmplx_smooth.columns = pd.MultiIndex.from_tuples([("Frequency", 'THz'), ('Smoothed Real Permittivity', 'dimensionless'), ('Smoothed Imaginary Permittivity', 'dimensionless')])  
                df_cmplx_smooth.set_index(df_perm.columns[0], drop=True, inplace=True)
                df_all_smooth = pd.concat([df_all_smooth,df_cmplx_smooth], axis=1)

        else:
                pass

        if Smooth:
            df_all_smooth.to_csv(filename, header=True, index=False, sep="\t")
        else:
            df_all_spec.to_csv(filename, header=True, index=False, sep="\t")
    
    elif Type == 'Time':
    
        print("need to sort this out")

    else:
         
         print("need to decide if you want to output frequency or time data to file, no data has been exported")

    return


