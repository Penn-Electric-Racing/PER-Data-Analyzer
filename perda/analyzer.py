from .csvparser import csvparser
from .dataplotter import dataplotter
from .arrayoperator import arrayoperator

class analyzer:
    def __init__(self):
        """Initialize a new analyzer instance for car data analysis.
       
        Creates instances of csvparser, dataplotter, and arrayoperator for data processing.
        """
        self.__csvparser = csvparser()
        self.__dataplotter = dataplotter()
        self.__operator = arrayoperator()
        self.__file_read = False

        print("Analyzer Created")

    def reset(self):
        """Reset the analyzer to its initial state.
       
        Clears all loaded data and resets internal components.
        Must be called before reading a new CSV file.
        """
        self.__csvparser.reset()
        self.__dataplotter.reset()
        self.__operator.reset()
        self.__file_read = False

        print("Reset Analyzer")

    def read_csv(self, path: str, input_align_name = "default"):
        """Read and parse a CSV file containing car data.
       
        Args:
            path (str): Path to the CSV file to read
            input_align_name (str, optional): Name of the variable to use for timestamp alignment.
                Defaults to "default" which uses the highest frequency data.
        """
        # if self.__file_read:
        #     raise AttributeError("Call .reset() before reading new csv")
        self.reset()
        self.__csvparser.read_csv(path, input_align_name)
        self.__dataplotter.set_csvparser(self.__csvparser)
        self.__operator.set_csvparser(self.__csvparser)
        self.__file_read = True

    def plot(self, variables: list, start_time = 0, end_time = -1, same_graph = True, time_unit = "s", h_lines: list = [], v_lines: list = []):
        """Plot data variables on a graph.

        Use plot when you want to visualize one or more variables over time on the same or separate graphs.
       
        Args:
            variables (list): List of variable names or arrays to plot
            start_time (int, optional): Start time for the plot. Defaults to 0.
            end_time (int, optional): End time for the plot. -1 for end of data. Defaults to -1.
            same_graph (bool, optional): Whether to plot all variables on same graph. Defaults to True.
            time_unit (str, optional): Time unit ("s" or "ms"). Defaults to "s".
            h_lines (list, optional): List of y-values to draw horizontal lines at. Defaults to empty list.
            v_lines (list, optional): List of x-values to draw vertical lines at. Defaults to empty list.
        """
        self.__dataplotter.set_plot(start_time, end_time, same_graph, time_unit)
        self.__dataplotter.plot_norm(variables, h_lines, v_lines)
    
    def plot_dual(self, variables: list, start_time = 0, end_time = -1, same_graph = True, time_unit = "s", h_lines: list = [], v_lines: list = []):
        """Plot data variables with dual y-axes.

        Use plot_dual when you want to visualize pairs of variables over time with separate y-axes.
        Each pair of variables will be plotted on its own graph with the first variable using the left y-axis
        and the second variable using the right y-axis.

        Args:
            variables (list): List of variable names or arrays to plot in pairs
            start_time (int, optional): Start time for the plot. Defaults to 0.
            end_time (int, optional): End time for the plot. -1 for end of data. Defaults to -1.
            same_graph (bool, optional): Whether to plot all pairs on same graph. Defaults to True.
            time_unit (str, optional): Time unit ("s" or "ms"). Defaults to "s".
            h_lines (list, optional): List of y-values to draw horizontal lines at. Defaults to empty list.
            v_lines (list, optional): List of x-values to draw vertical lines at. Defaults to empty list.
        """
        self.__dataplotter.set_plot(start_time, end_time, same_graph, time_unit)
        self.__dataplotter.plot_dual(variables, h_lines, v_lines)

    def get_nparray(self, short_name: str):
        """Get numpy array for a specific variable.
       
        Args:
            short_name (str): Name of the variable to retrieve
           
        Returns:
            numpy.ndarray: Array containing timestamp and value data
        """
        return self.__csvparser.get_np_array(short_name)
    
    def get_filtered_nparray(self, arr, start_time = 0, end_time = -1, time_unit = "s"):
        """Get numpy array filtered for a specific time range.
       
        Args:
            arr: Variable name or numpy array to filter
            start_time (int, optional): Start time for filtering. Defaults to 0.
            end_time (int, optional): End time for filtering. -1 for end of data. Defaults to -1.
            time_unit (str, optional): Time unit ("s" or "ms"). Defaults to "s".
           
        Returns:
            numpy.ndarray: Filtered array containing data within specified time range
        """
        return self.__operator.get_filtered_nparr(arr, start_time, end_time, time_unit)
    
    def filter_all_data(self):
        """Filter outliers from all data variables.
        
        This function filters outliers from all variables in the dataset using a rolling window approach.
        Outliers are identified using the interquartile range (IQR) method.
        
        Warning:
            This is an expensive operation that can take a significant amount of time, especially for large datasets.
            The function will prompt for confirmation before proceeding.
        """
        return self.__csvparser.filter_all_data()

    def align_array(self, align_list: list, match_type: str = "connect", start_time = 0, end_time = -1, time_unit = "s"):
        """Align multiple arrays based on timestamps.
       
        Args:
            align_list (list): List of variable names or arrays to align
            match_type (str, optional):  The method used for filling missing values.
                    "connect" - linear interpolation between nearest valid neighbors (default).
                    "extend_forward" - use the last valid value.
                    "extend_back" - use the next valid value. 
            start_time (int, optional): Start time for alignment. Defaults to 0.
            end_time (int, optional): End time for alignment. -1 for end of data. Defaults to -1.
            time_unit (str, optional): Time unit ("s" or "ms"). Defaults to "s".
           
        Returns:
            list: List of aligned arrays
        """
        return self.__operator.align_arrays(align_list, match_type, start_time, end_time, time_unit)

    def get_compute_arrays(self, op_list: list[str], match_type: str = "connect", start_time = 0, end_time = -1, time_unit = "s"):
        """Perform mathematical operations on arrays.
       
        Args:
            op_list (list[str]): List alternating between variable names and operators (+, -, *, /)
            match_type (str, optional): The method used for filling missing values.
                    "connect" - linear interpolation between nearest valid neighbors (default).
                    "extend_forward" - use the last valid value.
                    "extend_back" - use the next valid value. 
            start_time (int, optional): Start time for computation. Defaults to 0.
            end_time (int, optional): End time for computation. -1 for end of data. Defaults to -1.
            time_unit (str, optional): Time unit ("s" or "ms"). Defaults to "s".
           
        Returns:
            numpy.ndarray: Result of the mathematical operations
        """
        return self.__operator.get_compute_arrays(op_list, match_type, start_time, end_time, time_unit)
    
    def get_band_filter(self, band_list: list, sample_frequency = -1, filtered_lower_band = -1, filtered_upper_band = -1,
                        match_type: str = "connect", start_time = 0, end_time = -1, time_unit = "s"):
        """Apply band-pass filtering to signals.
        
        This function filters signals by removing frequency components outside a specified band.
        It first samples the signals at the given frequency, then applies FFT, removes unwanted 
        frequencies, and reconstructs the filtered signal via inverse FFT.

        Args:
            band_list (list): List of variable names or arrays to filter
            sample_frequency (int, optional): Sampling frequency in Hz. If -1, calculated from data. Defaults to -1.
            filtered_lower_band (int, optional): Lower cutoff frequency in Hz. If -1, defaults to fs/3. Defaults to -1.
            filtered_upper_band (int, optional): Upper cutoff frequency in Hz. If -1, no upper limit. Defaults to -1.
            match_type (str, optional): Method for handling missing values:
                    "connect" - linear interpolation between nearest neighbors (default)
                    "extend_forward" - use last valid value
                    "extend_back" - use next valid value
            start_time (int, optional): Start time for filtering. Defaults to 0.
            end_time (int, optional): End time for filtering. -1 for end of data. Defaults to -1.
            time_unit (str, optional): Time unit ("s" or "ms"). Defaults to "s".

        Returns:
            list: List of filtered arrays
        """
        return self.__operator.get_band_filter(band_list, sample_frequency, filtered_lower_band, filtered_upper_band, match_type, start_time, end_time, time_unit)

    
    def get_integral(self, arr, start_time = 0, end_time = -1, time_unit = "s"):
        """Calculate integral of data over time.
       
        Args:
            arr: Variable name or numpy array to integrate
            start_time (int, optional): Start time for integration. Defaults to 0.
            end_time (int, optional): End time for integration. -1 for end of data. Defaults to -1.
            time_unit (str, optional): Time unit ("s" or "ms"). Defaults to "s".
           
        Returns:
            float: Integral value
        """
        filtered_arr = self.__operator.get_filtered_nparr(arr, start_time, end_time, time_unit)
        integral, _ = self.__operator.get_integral_avg(filtered_arr)
        if time_unit == "s":
            integral /= 1e3
        return integral
    
    def get_average(self, arr, start_time = 0, end_time = -1, time_unit = "s"):
        """Calculate average value of data.
       
        Args:
            arr: Variable name or numpy array to average
            start_time (int, optional): Start time for averaging. Defaults to 0.
            end_time (int, optional): End time for averaging. -1 for end of data. Defaults to -1.
            time_unit (str, optional): Time unit ("s" or "ms"). Defaults to "s".
           
        Returns:
            float: Average value
        """
        filtered_arr = self.__operator.get_filtered_nparr(arr, start_time, end_time, time_unit)
        _, average = self.__operator.get_integral_avg(filtered_arr)
        return average
    
    def analyze_data(self, arr_list: list, start_time = 0, end_time = -1, time_unit = "s"):
        """Generate comprehensive statistics for given variables.
       
        Args:
            arr_list (list): List of variable names or arrays to analyze
            start_time (int, optional): Start time for analysis. Defaults to 0.
            end_time (int, optional): End time for analysis. -1 for end of data. Defaults to -1.
            time_unit (str, optional): Time unit ("s" or "ms"). Defaults to "s".
           
        Prints statistics including:
        - Data amount
        - Time range and duration
        - Maximum and minimum values with timestamps
        - Average value
        - Integral value
        """
        arr_num = 0
        for arr in arr_list:
            if type(arr) is str:
                var_name = arr
                canid = self.__csvparser.get_can_id(arr)
            else:
                var_name = f"Input Data {arr_num}"
                canid = -1

            filtered_arr = self.__operator.get_filtered_nparr(arr, start_time, end_time, time_unit)
            num_data = len(filtered_arr)
            data_start_time = filtered_arr[0,0]
            data_end_time = filtered_arr[-1,0]
            duration = data_end_time - data_start_time
            integral, average = self.__operator.get_integral_avg(filtered_arr)
            max_value, max_timestamp, min_value, min_timestamp = self.__operator.get_max_min(filtered_arr)
            decimals = 0

            if time_unit == "s":
                max_timestamp /= 1e3
                min_timestamp /= 1e3
                data_start_time /= 1e3
                data_end_time /= 1e3
                duration /= 1e3
                integral /= 1e3
                decimals = 3

            print(f"Statistics for **{var_name}**")
            if canid != -1:
                print(f"Can ID: {canid}")
            print(f"Data amount: {num_data}")
            display_start_time = format(data_start_time, f",.{decimals}f")
            display_end_time = format(data_end_time, f",.{decimals}f")
            duration = format(duration, f",.{decimals}f")
            max_time = format(max_timestamp,f",.{decimals}f")
            min_time = format(min_timestamp,f",.{decimals}f")
            print(f"Start: {display_start_time}{time_unit} | End: {display_end_time}{time_unit} | Duration: {duration}{time_unit}")
            print(f"Max Value: {max_value} ({max_time}{time_unit})")
            print(f"Min Value: {min_value} ({min_time}{time_unit})")
            print(f"Average: {average}")
            print(f"Integral: {integral}")
            print("\n")
    
    def calculate0to60(self, numWheels = 1):
        """Calculate and plot 0-60 mph acceleration data.
       
        Args:
            numWheels (int, optional): Number of wheels to consider (1-4). Defaults to 1.
           
        Raises:
            AttributeError: If numWheels is not between 1 and 4
        """
        if numWheels < 1 or numWheels > 4:
            raise AttributeError("One to four wheels. Default is 1")
        self.__dataplotter.plot0to60(numWheels)