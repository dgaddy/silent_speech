import time
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import scipy.signal

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, IpProtocolType
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

def remove_drift(signal, fs):
    b, a = scipy.signal.butter(3, 2, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)

def notch(signal, freq, sample_frequency):
    b, a = scipy.signal.iirnotch(freq, 30, sample_frequency)
    return scipy.signal.filtfilt(b, a, signal)

def notch_harmonics(signal, freq, sample_frequency):
    for f in range(freq, sample_frequency//2, freq):
        signal = notch(signal, f, sample_frequency)
    return signal

def filter_signal(signals, fs):
    """ signals is 2d: time, channels """
    result = np.zeros_like(signals)
    for i in range(signals.shape[1]):
        x = signals[:,i]
        x = notch_harmonics(x, 60, fs)
        x = remove_drift(x, fs)
        result[:,i] = x
    return result

def get_last_sequence(chunk_list, n, k, do_filtering, fs):
    cumulative_size = 0
    selected_chunks = [np.zeros((0,k))]
    for chunk in reversed(chunk_list):
        selected_chunks.append(chunk)
        cumulative_size += chunk.shape[0]
        if cumulative_size > n:
            break
    selected_chunks.reverse()
    result = np.concatenate(selected_chunks, 0)[-n:,:]
    if do_filtering and result.shape[0] > 12:
        result = filter_signal(result, fs)

    if result.shape[0] < n:
        result_padded = np.concatenate([np.zeros((n-result.shape[0],result.shape[1])), result], 0)
    else:
        result_padded = result
    return result_padded

class Recorder(object):
    def __init__(self, debug=False, display=True, num_channels=None, wifi=True):
        # make audio stream

        self.audio_stream = sd.InputStream(device=None, channels=1, samplerate=16000)

        # make emg stream

        params = BrainFlowInputParams()
        if debug:
            board_id = -1 # synthetic
            sample_rate = 256
        elif not wifi:
            board_id = BoardIds.CYTON_BOARD.value
            params.serial_port = '/dev/ttyUSB0'
            sample_rate = 250
        else:
            board_id = BoardIds.CYTON_WIFI_BOARD.value
            params.ip_port = 8001
            params.ip_address = '192.168.4.1'
            sample_rate = 1000
        self.emg_channels = BoardShim.get_emg_channels(board_id)
        if num_channels is not None:
            self.emg_channels = self.emg_channels[:num_channels]

        board = BoardShim(board_id, params)
        board.prepare_session()
        board.config_board('/3') # configure for digital read
        board.start_stream()
        self.board = board

        # config and make data holders

        audio_multiplier = int(16000/sample_rate)
        window = sample_rate*5

        self.audio_data = []
        self.emg_data = []
        self.button_data = []

        self.debug = debug
        self.previous_sample_number = -1

        # plot setup

        self.display = display
        if display:
            plt.ion()
            fig, (audio_ax, emg_ax) = plt.subplots(2)
            audio_ax.axis((0, window*audio_multiplier, -1, 1))
            emg_ax.axis((0, window, -300, 300))
            audio_lines = audio_ax.plot(np.zeros(window*audio_multiplier))
            emg_lines = emg_ax.plot(np.zeros((window,len(self.emg_channels))))
            for l,c in zip(emg_lines, ['grey', 'mediumpurple', 'blue', 'green', 'yellow', 'orange', 'red', 'sienna']):
                l.set_color(c)
            text = emg_ax.text(50,-250,'RMS: 0')

            for ax in (audio_ax, emg_ax):
                ax.set_yticks([0])
                ax.yaxis.grid(True)
                ax.tick_params(bottom=False, top=False, labelbottom=False,
                               right=False, left=False, labelleft=False)
            fig.tight_layout(pad=0)

            def update_plot(frame):
                """ This is called by matplotlib for each plot update. """
                audio_to_plot = get_last_sequence(self.audio_data, window*audio_multiplier, 1, False, sample_rate)
                audio_to_plot = audio_to_plot.squeeze(1)
                audio_lines[0].set_ydata(audio_to_plot)

                emg_to_plot = get_last_sequence(self.emg_data, window, len(self.emg_channels), True, sample_rate)
                for column, line in enumerate(emg_lines):
                    line.set_ydata(emg_to_plot[:, column])
                text.set_text('RMS: '+str(emg_to_plot[-sample_rate*2:-sample_rate//2].std()))
                return audio_lines + emg_lines

            self.ani = FuncAnimation(fig, update_plot, interval=30)

    def update(self):
        if self.display:
            # next two lines seem to be a better alternative to plt.pause(0.005)
            # https://github.com/matplotlib/matplotlib/issues/11131
            plt.gcf().canvas.draw_idle()
            plt.gcf().canvas.start_event_loop(0.005)
        else:
            time.sleep(0.005)

        current_audio = []
        while self.audio_stream.read_available > 0: # because stream.read_available seems to max out, leading us to not read enough with one read
            data, overflowed = self.audio_stream.read(self.audio_stream.read_available)
            assert not overflowed
            current_audio.append(data)
        if len(current_audio) > 0:
            self.audio_data.append(np.concatenate(current_audio,0))
            data = self.board.get_board_data() # get all data and remove it from internal buffer
            self.emg_data.append(data[self.emg_channels,:].T)

            if not self.debug:
                for sn in data[0,:]:
                    if self.previous_sample_number != -1 and sn != (self.previous_sample_number+1)%256:
                        print(f'skip from {self.previous_sample_number} to {sn}')
                    self.previous_sample_number = sn

                is_digital_inputs = data[12,:] == 193
                button_data = data[16,is_digital_inputs].astype(np.bool)
                self.button_data.append(button_data)
                if sum(button_data) != 0:
                    print('button pressed')

    def get_data(self):
        emg = np.concatenate(self.emg_data, 0)
        audio = np.concatenate(self.audio_data, 0).squeeze(1)
        button = np.concatenate(self.button_data, 0)
        chunk_sizes = [(e.shape[0],a.shape[0],b.shape[0]) for e, a, b in zip(self.emg_data, self.audio_data, self.button_data)]
        self.emg_data = []
        self.audio_data = []
        self.button_data = []
        return emg, audio, button, chunk_sizes

    def __enter__(self):
        self.audio_stream.start()
        return self

    def __exit__(self, type, value, traceback):
        self.audio_stream.stop()
        self.audio_stream.close()

        self.board.stop_stream()
        self.board.release_session()

        plt.close()

if __name__ == '__main__':
    with Recorder(debug=False, wifi=True, num_channels=1) as r:
        while True:
            r.update()
