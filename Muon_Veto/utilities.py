# ==================== #
#       IMPORTS        #
# ==================== #
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def calc_rate(directory):
    """
    Calculates measured and corrected rates for data runs in a specified directory.

    The corrected rate accounts for events lost due to buffer overflows.

    Args:
        directory (str): The full path to the directory containing the run data files.

    Returns:
        dict: A dictionary where keys are run numbers and values are dictionaries
              containing rates, efficiency, run type, and status.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found: {directory}")
        return {}

    file_prefix = 'muons_run_'
    file_suffix = '.h5'
    hdf5_key = 'data'
    timestamp_column = 'timestamp'
    lost_buffer_column = 'lostBuffer' # Columna con los eventos perdidos
    run_results = {}

    all_files = os.listdir(directory)
    
    run_files_info = []
    for filename in all_files:
        if filename.startswith(file_prefix) and filename.endswith(file_suffix):
            match = re.search(r'muons_run_(\d+)\.h5', filename)
            if match:
                run_number = int(match.group(1))
                run_type = 'No Source' if run_number % 2 != 0 else 'With Source'
                run_files_info.append({'filename': filename, 'run_number': run_number, 'type': run_type})

    run_files_info.sort(key=lambda x: x['run_number'])

    if not run_files_info:
        print(f"Warning: No valid run files found in {directory}.")
        return {}
        
    for file_info in run_files_info:
        run_num = file_info['run_number']
        file_name = file_info['filename']
        run_type = file_info['type']
        file_path = os.path.join(directory, file_name)

        try:
            data_df = pd.read_hdf(file_path, hdf5_key)

            if data_df.empty or timestamp_column not in data_df.columns:
                status_msg = 'Empty DataFrame or Timestamp Column Missing'
                run_results[run_num] = {'Run Type': run_type, 'Status': status_msg}
                continue

            # --- LÓGICA DE CORRECCIÓN AÑADIDA ---
            if lost_buffer_column not in data_df.columns:
                status_msg = f"Column '{lost_buffer_column}' Not Found"
                run_results[run_num] = {'Run Type': run_type, 'Status': status_msg}
                continue

            first_timestamp = data_df[timestamp_column].iloc[0]
            last_timestamp = data_df[timestamp_column].iloc[-1]
            time_difference = (last_timestamp - first_timestamp) * 0.001 # a segundos

            if time_difference > 0:
                events_registered = len(data_df)
                events_lost = data_df[lost_buffer_column].sum()
                total_events_estimated = events_registered + events_lost
                
                # Calcular tasas y eficiencia
                measured_rate = events_registered / time_difference
                corrected_rate = total_events_estimated / time_difference
                daq_efficiency = (events_registered / total_events_estimated) if total_events_estimated > 0 else 1.0

                run_results[run_num] = {
                    'Measured Rate (Hz)': f'{measured_rate:.6f}',
                    'Corrected Rate (Hz)': f'{corrected_rate:.6f}',
                    'DAQ Efficiency': f'{daq_efficiency:.4f}',
                    'Events Registered': events_registered,
                    'Events Lost': int(events_lost),
                    'Run Type': run_type, 
                    'Status': 'Success'
                }
            else:
                run_results[run_num] = {
                    'Run Type': run_type, 
                    'Status': 'Zero/Negative Time Diff'
                }

        except KeyError:
            run_results[run_num] = {'Run Type': run_type, 'Status': f"HDF5 Key '{hdf5_key}' Not Found"}
        except Exception as e:
            run_results[run_num] = {'Run Type': run_type, 'Status': f'Error: {e}'}

    return run_results

def plot_rates(data_df, skip_bars=None):
    """
    Creates a scatter plot of the corrected rates, grouped by run type.

    Args:
        data_df (pd.DataFrame): The DataFrame containing the consolidated rates.
        skip_bars (list, optional): List of bar labels to exclude, e.g. ["B2", "C5"].
    """
    if data_df.empty:
        print("Cannot plot: The input DataFrame is empty.")
        return

    # Usar la columna con la tasa corregida para el ploteo
    rate_column_to_plot = 'Corrected Rate (Hz)'

    # Filtrar solo filas exitosas y con un valor de tasa válido
    plot_data = data_df[data_df['Status'] == 'Success'].copy()
    plot_data.dropna(subset=[rate_column_to_plot], inplace=True) # Eliminar filas donde la tasa es N/A

    if plot_data.empty:
        print("No successful rates to plot.")
        return

    if skip_bars:
        plot_data = plot_data[~plot_data['Bar Label'].isin(skip_bars)]

    plot_data = plot_data.sort_values(by=['Bar Letter', 'Bar Number'])

    no_source_data = plot_data[plot_data['Run Type'] == 'No Source']
    with_source_data = plot_data[plot_data['Run Type'] == 'With Source']

    plt.style.use('ggplot')
    plt.figure(figsize=(14, 8))

    plt.scatter(
        no_source_data['Bar Label'],
        no_source_data[rate_column_to_plot],
        color='blue',
        marker='o',
        label='Without Source'
    )

    plt.scatter(
        with_source_data['Bar Label'],
        with_source_data[rate_column_to_plot],
        color='red',
        marker='^',
        label='With Source'
    )

    plt.title('Muon Veto Corrected Rates for All Bars')
    plt.xlabel('Bar')
    plt.ylabel('Corrected Rate (Hz)')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.show()

base_directory = '/lhome/ific/c/ccortesp/Analysis/Muon_Veto/data/'
final_rates_list = []
    
for root, dirs, files in os.walk(base_directory):
    if os.path.basename(os.path.dirname(root)) in ['A', 'B', 'C', 'D', 'E', 'F']:
        bar_letter = os.path.basename(os.path.dirname(root))
        bar_number = os.path.basename(root)

        if bar_letter not in ['A', 'B', 'C', 'D', 'E', 'F'] or not bar_number.isdigit():
            continue
            
        rates_dict = calc_rate(root)
            
        # Desempaquetar el diccionario con los nuevos campos
        for run_num, data in rates_dict.items():
            run_entry = {
                'Bar Letter': bar_letter,
                'Bar Number': int(bar_number),
                'Bar Label': f"{bar_letter}{bar_number}",
                'Run Number': run_num,
                'Run Type': data.get('Run Type'),
                'Status': data.get('Status'),
                # Usamos .get() para evitar errores si una corrida falló y no tiene estas claves
                'Measured Rate (Hz)': data.get('Measured Rate (Hz)'),
                'Corrected Rate (Hz)': data.get('Corrected Rate (Hz)'),
                'DAQ Efficiency': data.get('DAQ Efficiency'),
                'Events Registered': data.get('Events Registered'),
                'Events Lost': data.get('Events Lost')
            }
            final_rates_list.append(run_entry)
                
# Crear el DataFrame final
if final_rates_list:
    final_df = pd.DataFrame(final_rates_list)
    # Convertir columnas numéricas, usando 'coerce' para manejar valores N/A o faltantes
    numeric_cols = [
        'Run Number', 'Measured Rate (Hz)', 'Corrected Rate (Hz)', 
        'DAQ Efficiency', 'Events Registered', 'Events Lost'
    ]
    for col in numeric_cols:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
    
    # Imprime el nuevo DataFrame para verificar. Verás las nuevas columnas.
    # print(final_df.sort_values(by=['Bar Letter', 'Bar Number', 'Run Number']).reset_index(drop=True))
else:
    print("No valid run data found in the specified directories.")
    final_df = pd.DataFrame() # Crear un DataFrame vacío para evitar errores posteriores




def COMPUTE_RATE_FILE(FILE_PATH):
    """
    Helper function to calculate the event rate from a single HDF5 file.

    This function reads an HDF5 file containing event data, computes the event rate 
    (events per second), and calculates the associated error. The event rate is 
    determined based on the timestamps of the events in the file.

    Parameters:
    -----------
    FILE_PATH : str
        Path to the HDF5 file containing the event data.

    Returns:
    --------
    tuple
        A tuple containing:
        - Rate_CV (float): The calculated event rate (events per second).
        - Rate_Err (float): The error in the calculated event rate.
        If the time difference is invalid or an error occurs, returns (-1, -1).
    """
    # HDF5 key to access the data
    h5_key = 'data'

    try:
        # Read the HDF5 file into a DataFrame
        data_df = pd.read_hdf(FILE_PATH, key=h5_key)

        # Check if the DataFrame is not empty and contains the 'timestamp' column
        if not data_df.empty and 'timestamp' in data_df.columns:
                
            # Sort the DataFrame by timestamp to ensure correct chronological order
            data_df = data_df.sort_values(by='timestamp').reset_index(drop=True)
            
            # Extract the first and last timestamps
            first_timestamp = data_df['timestamp'].iloc[0]
            last_timestamp = data_df['timestamp'].iloc[-1]
            
            time_difference = (last_timestamp - first_timestamp) * 0.001        # in [s]

            # Calculate the event count and its error
            evt_CV = len(data_df)
            evt_err = np.sqrt(evt_CV)

            # Ensure the time difference is valid before calculating the rate
            if time_difference > 0:
                Rate_CV  = evt_CV / time_difference
                Rate_Err = evt_err / time_difference

                return Rate_CV, Rate_Err
            else:
                # Invalid time difference, return error values
                return -1, -1

    # Handle specific errors
    except KeyError:
        print(f"Warning: HDF5 key '{h5_key}' not found in file {FILE_PATH}. Skipping this file.")
    except Exception as e:
        print(f"An error occurred while processing {FILE_PATH}: {e}")

    # Return error values in case of failure
    return -1, -1

def PLOT_CHARGE_HIST_FILE(FILE_PATH, PLOT_LABEL, SIGNAL_CHANNELS=[6, 7], X_LIMITS=[0, 2000]):
    """
    Helper function to plot the charge histogram from a single HDF5 file.

    This function reads charge data from an HDF5 file and generates a histogram 
    for the charge distribution of specified channels. The histogram includes 
    a control/noise channel and two active signal channels. The function also 
    allows setting custom x-axis limits for the histogram.

    Parameters:
    -----------
    FILE_PATH : str
        Path to the HDF5 file containing the charge data.
    PLOT_LABEL : str
        Label for the plot title, typically describing the dataset or file.
    SIGNAL_CHANNELS : list of int, optional
        List of two integers specifying the indices of the active signal channels 
        to be plotted. Default is [6, 7].
    X_LIMITS : list of int, optional
        List of two integers specifying the x-axis limits for the histogram. 
        Default is [0, 2000].

    Returns:
    --------
    None
        The function generates and displays a histogram plot. It does not return any value.
    """
    try:
        # Open the HDF5 file in read mode
        with h5py.File(FILE_PATH, "r") as f:

            # Check if the 'charges' dataset exists in the file
            if 'charges' not in f:
                print(f"Warning: 'charges' dataset not found in {FILE_PATH}.")
                return

            # Read the 'charges' dataset into a NumPy array
            charges_df = f["charges"][:]
                
            # Extract charge data for the control/noise channel and the two signal channels
            charge_control = charges_df[:, 5]
            charge_signal_1 = charges_df[:, SIGNAL_CHANNELS[0]]
            charge_signal_2 = charges_df[:, SIGNAL_CHANNELS[1]]  # Second signal channel

            # ----- Plotting ----- #
            # Define histogram bins with a fixed width of 10 ADC counts
            bins = np.arange(X_LIMITS[0], X_LIMITS[1] + 10, 10)

            # Set the plot style
            plt.style.use('ggplot')
            plt.figure(figsize=(10, 6))    
            
            # Plot histograms for each channel
            plt.hist(charge_control,  bins=bins, histtype='step', color='red',   linewidth=1.0, alpha=1.0, label='Noise')
            plt.hist(charge_signal_1, bins=bins, histtype='step', color='blue',  linewidth=1.0, alpha=1.0, label=f'Channel {SIGNAL_CHANNELS[0]} (Active)')
            plt.hist(charge_signal_2, bins=bins, histtype='step', color='green', linewidth=1.0, alpha=1.0, label=f'Channel {SIGNAL_CHANNELS[1]} (Active)')

            # Add labels, title, and legend
            plt.xlabel('Charge (ADC)')
            plt.xlim(X_LIMITS)
            plt.ylabel('Counts')
            plt.yscale('log')
            plt.legend()
            plt.title(f'{PLOT_LABEL}')
            
            # Add grid and adjust layout
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()

    except Exception as e:
        # Handle any exceptions that occur during file reading or plotting
        print(f"An error occurred while processing {FILE_PATH}: {e}")