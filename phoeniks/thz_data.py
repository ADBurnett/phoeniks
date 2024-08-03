import numpy as np
from scipy import interpolate
from scipy.fft import next_fast_len
from scipy import signal as sp
from scipy.constants import c as c_0
from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def interpolate_fd(frequency, new_frequency, trace_std):
    # Interpol1d only works with real numbers, needs to split complex number in real and imag
    f_interpol_real = interpolate.interp1d(frequency, trace_std.real)
    f_interpol_imag = interpolate.interp1d(frequency, trace_std.imag)
    new_std_real = f_interpol_real(new_frequency)
    new_std_iamg = f_interpol_imag(new_frequency)
    return new_std_real + 1j * new_std_iamg


class Data:
    """This class contains all THz data in time and frequency domain for the cases of
    1. Dark measurement (THz blocked)
    2. Reference measurement (without sample)
    3. Sample measurement (with sample).

    The analysis can be done with different amounts of input data:

    1. The user provide time-domain data of reference,
    sample and dark measurements (averaged over multiple measurements) as well as standard deviations (based on the
    single measurements) in frequency-domain of the three measurements. This allows to use the SVMA-filter later and
    is the best case to use this program.

    2. The user provides reference, sample and dark measurement (averaged) in time domain. The dark trace allows to
    quantify the dynamic range for all frequencies. THe SVMA-filter cannot be used.

    3. The user provides only reference and sample trace (averaged) in time domain. The user can supply an upper
    frequency, from which the noise floor is determined. Can lead to wrong results (especially if data is taken from a
    Lock-In amplifier with an integrated low-pass filter, which leads to a non-flat noise floor)

    available_modes = ["reference_sample_dark_standard_deviations",
                       "reference_sample_dark",
                       "reference_sample"]"""
    
 



    def __init__(self,
                 time,
                 td_reference,
                 td_sample,
                 thickness,
                 td_dark=None,
                 td_ref_std=None,
                 td_samp_std=None,
                 td_dark_std=None,
                 fd_reference_std=None,
                 fd_sample_std=None,
                 fd_dark_std=None) -> None:
        # Define instance variables in time domain
        self.time = time
        self.td_reference = td_reference
        self.td_sample = td_sample
        self.td_dark = td_dark
        self.time_raw = time
        self.tdraw_reference = td_reference
        self.tdraw_sample = td_sample
        self.tdraw_dark = td_dark
        self.td_ref_std = td_ref_std
        self.td_samp_std = td_samp_std
        self.td_dark_std = td_dark_std
        self.max_ref = np.argmax(self.td_reference)
        self.max_samp = np.argmax(td_sample)
        self.shift = int((self.max_samp - self.max_ref))
        self.dt = (self.time[-1] - self.time[0]) / (len(self.time) - 1)
        self.sampling_rate = 1 / self.dt
        # Define instance variables in frequency domain
        self.fd_reference_std = fd_reference_std
        self.fd_sample_std = fd_sample_std
        self.fd_dark_std = fd_dark_std
        # thickness should be in meters
        self.thickness = thickness
        # Variables for later extraction
        self.frequency = np.fft.rfftfreq(len(self.time), self.dt)
        self.frequency_raw = self.frequency
        self.omega = 2 * np.pi * self.frequency
        self.phase_reference = None
        self.phase_sample = None
        self.n = None
        self.k = None
        self.alpha = None
        self.delta_max = None
        # Depending, how much data is provided, the right mode will be selected
        self.mode = "reference_sample_dark_standard_deviations"
        if fd_reference_std is None or fd_sample_std is None or fd_dark_std is None:
            self.mode = "reference_sample_dark"
        else:
            self.fd_reference_std = fd_reference_std
            self.fd_sample_std = fd_sample_std
            self.fd_dark_std = fd_dark_std
            self.fd_reference_std_raw = self.fd_reference_std
            self.fd_sample_std_raw = self.fd_sample_std
            self.fd_dark_std_raw = self.fd_dark_std
        if td_dark is None:
            self.mode = "reference_sample"
        else:
            self.fd_dark = np.fft.rfft(self.td_dark)
            if len(self.fd_dark) != len(self.frequency):
                raise ValueError("Supplied frequency data does not match calculated frequency array. " +
                                 "Frequency data should be only defined for positive frequencies.")
        self.fd_reference = np.fft.rfft(self.td_reference)
        self.fd_sample = np.fft.rfft(self.td_sample)
        self.H = self.fd_sample / self.fd_reference
        self.phase_H = None
        self.H_approx = None
        if (len(self.fd_reference) != len(self.frequency)) or \
                (len(self.fd_sample) != len(self.frequency)):
            raise ValueError("Supplied frequency data does not match calculated frequency array. " +
                             "Frequency data should be only defined for positive frequencies.")
        
            # read Bootstrap external theme
        external_stylesheets = [dbc.themes.BOOTSTRAP]
        
        self.app = Dash(__name__, external_stylesheets=external_stylesheets)

    def offset_time_to_reference_peak(self) -> None:
        """Shift peak of reference trace to 0."""
        self.time -= self.time[self.max_ref]


    def pad_zeros(self,zero_padding):
        if not self._data_is_windowed():
            raise ValueError("Data is not windowed. The data must be windowed before zeros can be padded.")
        x = next_fast_len(zero_padding)
        if x < len(self.time_raw):
            x = next_fast_len(len(self.time_raw))         
                # perfrom the FFT
        self.fd_reference = np.fft.rfft(self.td_reference, x)
        self.fd_sample = np.fft.rfft(self.td_sample, x)
        if self.mode != "reference_sample":
            self.fd_dark = np.fft.rfft(self.td_dark, x)
        new_frequency = np.fft.rfftfreq(x, self.dt)
        if self.mode == "reference_sample_dark_standard_deviations":
            self.fd_reference_std = interpolate_fd(self.frequency_raw, new_frequency, self.fd_reference_std_raw)
            self.fd_sample_std = interpolate_fd(self.frequency_raw, new_frequency, self.fd_sample_std_raw)
            self.fd_dark_std = interpolate_fd(self.frequency_raw, new_frequency, self.fd_dark_std_raw)
        self.frequency = new_frequency
        self.omega = 2 * np.pi * self.frequency
        self.zero_padding = x

    def get_window(self, start, end, windows='tukey', alpha=0.16) -> np.ndarray:


        self.window_functions = {
            'boxcar': sp.windows.boxcar,
            'triang': sp.windows.triang,
            'blackman': sp.windows.blackman,
            'blackmanharris' : sp.windows.blackmanharris,
            'hamming': sp.windows.hamming,
            'general_hamming' : sp.windows.general_hamming,
            'hann': sp.windows.hann,
            'tukey': sp.windows.tukey,
            # Add other window types here
        }


        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("Start and end indices must be integers.")
        if start >= end:
            raise ValueError("Start index must be less than end index.")
        
        length = end - start

        window_func = self.window_functions.get(windows.lower())
        if window_func is None:
            raise ValueError(f"Invalid window type: {window}")

        try:
            window = np.zeros_like(self.tdraw_reference)

            if window_func is sp.windows.tukey:
                window[start:end] = window_func(length, alpha, sym=False)  
            elif window_func is sp.windows.general_hamming:
                window[start:end] = window_func(length, alpha, sym=False)              
            else:
                window[start:end] = window_func(length, sym=False)
       
        except ValueError as e:
            raise ValueError("Your window is larger than the data:", e)
        return window

    def window_traces(self, start: int, end: int, windows='tukey', alpha=0.16) -> None:
        """Windows time domain traces with smoothness-factor alpha according to an (asymmetric) Blackman window.

        alpha (float): smoothness-factor between 0 and 1.
        start (int): Time point, where the window starts to let through signal in [s].
        stop (int): Time point, where the window stops to let through signal in [s]."""
        # fix max heights of windows for plotting purposes
        self.window = self.get_window(start, end, windows, alpha)
        self.td_reference = self.tdraw_reference * self.window
        start += self.shift
        end += self.shift
        self.samp_window = self.get_window(start, end, windows, alpha)
        self.td_sample = self.tdraw_sample * self.samp_window
        if self.mode != "reference_sample":
            self.td_dark = self.tdraw_dark * self.window
            self.fd_dark = np.fft.rfft(self.td_dark)
        # Update frequency domain data
        self.fd_reference = np.fft.rfft(self.td_reference)
        self.fd_sample = np.fft.rfft(self.td_sample)
        self.windowed = True
        self.win_start = start
        self.win_end = end

    def _data_is_windowed(self) -> bool:
        """Check if the data is close to zero at beginning and end == is windowed."""
        if self.windowed == True:
            windowed = True
        else:

            if self.mode != "reference_sample":
                traces = [self.td_reference, self.td_sample, self.td_dark]
            else:
                traces = [self.td_reference, self.td_sample]
            for trace in traces:
                if not np.isclose(trace[0], 0):
                    windowed = False
                if not np.isclose(trace[-1], 0):
                    windowed = False
        return windowed

    def linear_offset(self, time_trace, idx_beginning=10, idx_end=-31):
        """Subtract offset from the beginning of the time trace and interpolates a line with the last data points.
         This line is then subtracted from the time trace.

         Input:
         time_trace (np.array, 1D, float) : Array containing the THz signal in time domain.
         idx_beginning (int) :              How many data samples from the beginning are taken to create an average
                                            and subtract it?
         idx_end (int, negative) :          How many data samples from the end (thus negative) should be taken,
                                            to create a linear fit and substract it from the data?

        Output:
        time_trace (np.array, 1D, float) : THz time data with linear offset correction.
         """
        # Original code:
        # ma = np.mean(rv[:10])
        # rv -= ma
        # Calculating average of the last 30 sampling points in reference trace
        # me = np.mean(rv[-30:])
        # Creating a linear function with the slope between beginning (which is already subtracted, so 0) and end.
        # It will automatically create a linear function,
        # which is defined from the average of the first 10 datapoints and the last 30 data points.
        # o1 = np.arange(n - 1) * me / (n - 1)
        # rv -= o1
        n = len(time_trace)
        time_trace -= np.mean(time_trace[:idx_beginning])
        linear_function = np.arange(n) * np.mean(time_trace[idx_end:]) / (n - 1)
        time_trace -= linear_function
        return time_trace
    

    def simple_extract(self, logarithm = 'natural'):


        self.log_functions = {
            'decadic': np.log10,
            'natural': np.log,
        }

        log_func = self.log_functions.get(logarithm.lower())
        if log_func is None:
            raise ValueError(f"Invalid logarithm type: {type}")
        
        n_air = 1.00027

        self.n = np.zeros(len(self.frequency))
        phase_diff = np.unwrap(np.angle(self.fd_reference)-np.angle(self.fd_sample))
        if self.omega[0] == 0:
            self.n[0] = 0
            self.n[1:] = ((c_0)/(self.omega[1:]*self.thickness)) * phase_diff[1:] + n_air;
        else:
            self.n = ((c_0)/(self.omega*self.thickness)) * phase_diff + n_air;
        
        # calculate some useful parameters
        amplitude_ratio = np.abs(self.fd_sample) / np.abs(self.fd_reference)
        self.k = np.zeros(len(self.frequency))
        t_co = 4*self.n*n_air/(self.n+n_air)**2
  

        if self.omega[0] == 0:
            # Circumvent division by 0 error
            self.k[0] = 0
            self.k[1:] = -c_0/self.thickness/self.omega[1:]*log_func(amplitude_ratio[1:]/t_co[1:])
        else:
            self.k = -c_0/self.thickness/self.omega*log_func(amplitude_ratio/t_co)
        self.a = self.k * 2 * self.omega / c_0;
    

        refractive_index = np.empty(self.n.shape, dtype=complex)
        refractive_index.real = self.n
        refractive_index.imag = self.k
        self.ecmplx = refractive_index * refractive_index
        self.ncmplx = refractive_index
        self.ereal = np.real(self.ecmplx)
        self.eimag = np.imag(self.ecmplx)
    

    # Functions needed for plotly processing app
    def generate_layout(self):
        # app layout using bootstrap
        
        self.app.layout = html.Div([
            dbc.Row([
                        dbc.Col([
                    dbc.Row(
                        dbc.Col(html.Div(html.H1('THz Spectral Processor', style={'textAlign': 'center'})))
                    ),
                    dbc.Row([
                        dbc.Col(
                            html.Div([
                                html.Label("Zero Padding", htmlFor="zero_fill"),
                                dcc.Slider(1024, 1E5, value=4096, id='zero_fill')
                                ]), width = 4
                                ),
                        dbc.Col(
                            html.Div([
                                html.Label("Window Curve", htmlFor="window_curve"),
                                dcc.Slider(0, 1,0.01, marks={(i/10): '{}'.format(i/10) for i in range(11)},
                                value=0.1,
                                id='window_curve')
                                ]) ,width = 4
                                ),
                        dbc.Col(
                            html.Div([
                                html.Label("Window Choice", htmlFor="window_choice"),
                                dcc.Dropdown(['boxcar', 'triang', 'blackman', 'blackmanharris', 'hamming', 'general_hamming', 'hann', 'tukey'], 
                                value='tukey',
                                id='window_choice')
                                ]) ,width = 4                        

                    ),
                    ]),
                    dbc.Row([
                        dbc.Col(
                            html.Div([
                                html.Label("Window Start and End", htmlFor="window_range"),
                                dcc.RangeSlider(0, len(self.td_sample),1,marks=None, value=[0,int(len(self.td_sample)*.9)],id='window_range'),
                            ]), width=12
                        ),
                    ]),
                    dbc.Row([
                        dbc.Col(html.Div(dcc.Graph(id='graph1')), width=12),
                    ]),
                    dbc.Row([
                        dbc.Col(html.Div(dcc.Graph(id='graph2')), width=6),
                        dbc.Col(html.Div(dcc.Graph(id='graph3')), width=6)
                    ]),
                    dbc.Row([
                        dbc.Col(html.Div(dcc.Graph(id='graph4')), width=6),
                        dbc.Col(html.Div(dcc.Graph(id='graph5')), width=6)
                    ])

                ], width=12)

            
            ])
        ])

            
        # callbacks from the slider updates
        @callback(
            Output('graph1', 'figure',allow_duplicate=False),
            Output('graph2', 'figure',allow_duplicate=False),
            Output('graph3', 'figure',allow_duplicate=False),
            Output('graph4', 'figure',allow_duplicate=False),
            Output('graph5', 'figure',allow_duplicate=False),
            Input('zero_fill', 'value'),
            Input('window_range', 'value'),
            Input('window_curve', 'value'),
            Input('window_choice', 'value'),
            prevent_initial_call=False
            )
        
      
        def update_figure(zp, rng, curve, window_choice):
            self.window_traces(start=rng[0], end=rng[1], windows=window_choice, alpha=curve)
            self.pad_zeros(zero_padding=zp)
            self.simple_extract()
            fig=self.window_fig(rng[0],rng[1])
            fig1=self.spectral_fig()
            fig2=self.phase_fig()
            fig3=self.abs_fig()
            fig4=self.ref_fig()
            return fig, fig1, fig2, fig3, fig4
        



        # Helper functions to define the curves

    def window_fig(self, start, stop):
        ''' generates a figure with the raw data with window superimposed'''
        fig = go.Figure()
        fig = make_subplots()
        fig.update_layout(title='Raw Data and Windows')
        fig.update_xaxes(title='Time (ps)')
        fig.update_yaxes(title='Amplitude (V)')
        fig.add_traces(go.Scatter(x=self.time*1E12, y=self.td_reference,name='ref', line_color='black'))
        fig.add_traces(go.Scatter(x=self.time*1E12, y=self.td_sample,name='sample', line_color='blue'))
        if 'dark' in self.mode:
            fig.add_traces(go.Scatter(x=self.time*1E12, y=self.td_dark,name='dark', line_color='green'))
        fig.add_traces(go.Scatter(x=self.time*1E12, y=(self.window * np.max(self.td_reference)), name='Ref_window', line_color='black'))
        fig.add_traces(go.Scatter(x=self.time*1E12, y=(self.samp_window * np.max(self.td_reference)), name='Samp_window', line_color='blue'))
        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
        #fig.update_layout(template="plotly_dark")
        return fig

    def spectral_fig(self):
        ''' generates the post fourier transformed spectrum plot'''
        fig = go.Figure()
        fig.update_layout(title='Amplitude of FFT (Zero-padded)')
        fig.update_xaxes(title='Frequency (THz)')
        fig.update_yaxes(title='Amplitude (a.u.)', type='log')
        fig.add_traces(go.Scatter(x=self.frequency/1E12, y=np.abs(self.fd_reference),name='ref', line_color='black'))
        fig.add_traces(go.Scatter(x=self.frequency/1E12, y=np.abs(self.fd_sample),name='sample', line_color='blue'))
        if 'dark' in self.mode:
            fig.add_traces(go.Scatter(x=self.frequency/1E12, y=np.abs(self.fd_dark),name='dark', line_color='green'))
        return fig

    def phase_fig(self):
        '''generates the phase plot'''
        fig = go.Figure()
        fig.update_layout(title='Phase of FFT (Zero-padded)')
        fig.update_xaxes(title='Frequency (THz)')
        fig.update_yaxes(title='Phase (radians)')
        fig.add_traces(go.Scatter(x=self.frequency/1E12, y=np.angle(self.fd_reference),name='ref', line_color='black'))
        fig.add_traces(go.Scatter(x=self.frequency/1E12, y=np.angle(self.fd_sample),name='sample', line_color='blue'))
        if 'dark' in self.mode:
            fig.add_traces(go.Scatter(x=self.frequency/1E12, y=np.angle(self.fd_dark),name='dark', line_color='green'))
        fig.add_traces(go.Scatter(x=self.frequency/1E12, y=np.unwrap(np.angle(self.fd_reference)),name='ref', line_color='black'))
        fig.add_traces(go.Scatter(x=self.frequency/1E12, y=np.unwrap(np.angle(self.fd_sample)),name='sample', line_color='blue'))
        if 'dark' in self.mode:
            fig.add_traces(go.Scatter(x=self.frequency/1E12, y=np.unwrap(np.angle(self.fd_dark)),name='dark', line_color='green'))  
        return fig

    def abs_fig(self):
        ''' generates the post fourier transformed spectrum plot'''
        fig = go.Figure()
        fig.update_layout(title='Absorption Coeffcient')
        fig.update_xaxes(title='Frequency (THz)')
        fig.update_yaxes(title='Absorption Coeffcient (m-1)')
        fig.add_traces(go.Scatter(x=self.frequency/1E12, y=self.a, name='Absorption', line_color='black'))
        return fig


    def ref_fig(self):
        ''' generates the post fourier transformed spectrum plot'''
        fig = go.Figure()
        fig.update_layout(title='Refractive Index')
        fig.update_xaxes(title='Frequency (THz)')
        fig.update_yaxes(title='Refractive Index')
        fig.add_traces(go.Scatter(x=self.frequency/1E12, y=self.n,name='refractive index', line_color='black'))
        return fig
