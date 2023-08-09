import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import math

# synthesizes simple ultrasound data of a sigle scatterer
# The scatterer is central to the transciever and a single planewave is assumed as the excitation
# returns a dictionary with the following keys:
#   pulse_channel_data: The channel data as it would have been recorded by the ultrasound system 
#                       (numpy array with shape [num_elements, num_samples])
#   pulse_focus: Data with focusing delays applied (aperture data) for a line central to the transceiver
#                (numpy array with shape [num_elements, num_samples])
#   pulse_side: Data with focusing delays applied (aperture data) for a line lateral to the transceiver 
#               (and consequently the scatterer). (numpy array with shape [num_elements, num_samples])
#   line_aperture_data: The aperture data for a number of scanlines at once, allows creation of an image
#                       (numpy array with shape [num_elements, num_samples, num_lines])
#   params: Dictionary with "ultrasound acquisition" parameters
def create_toy_data():
    num_elements = 40
    num_samples = 101
    num_samples = 171

    element_spacing = 0.1  # mm
    point_depth = 40  # mm
    speed_of_sound = 1540e3  # mm/s
    sampling_frequency = 200e6  # Hz

    num_lines = 128
    line_spacing = 0.05  # mm

    pulse_frequency = 7e6
    pulse_bandwidth = 0.40
    pulse_side_line_idx = 20
    # determined through:
    # pulse_side_ext = np.concatenate((np.zeros(pulse_side.shape), pulse_side, np.zeros(pulse_side.shape)), 1)
    # plt.plot([np.max(scipy.signal.convolve(pulse_side_ext, line_aperture_data[:, :, i], 'same').flatten()) for i in
    #           range(line_aperture_data.shape[-1])])
    # and selecting the index with the largest correlation value

    pulse_side_phase_total = 2 * math.pi

    def pulse(t_center):
        t = np.linspace(-num_samples / 2 / sampling_frequency - t_center,
                        num_samples / 2 / sampling_frequency - t_center, num_samples)
        return scipy.signal.gausspulse(t, fc=pulse_frequency, bw=pulse_bandwidth) * 1024

    pulse_focus = np.tile(pulse(0), [num_elements, 1])

    phases = np.linspace(-pulse_side_phase_total / 2, pulse_side_phase_total / 2, num_elements)
    t_centers = phases / (2.0 * math.pi * pulse_frequency)
    pulse_side = np.stack([pulse(t_center) for t_center in t_centers])

    element_positions = np.linspace(-num_elements / 2 * element_spacing,
                                    num_elements / 2 * element_spacing,
                                    num_elements)
    t_centers = np.sqrt(element_positions ** 2 + point_depth ** 2) / speed_of_sound
    t_base = min(t_centers)
    pulse_channel_data = np.stack([pulse(t_center - t_base) for t_center in t_centers])

    line_aperture_data = np.zeros(pulse_channel_data.shape + (num_lines,))
    line_offsets = np.zeros((num_lines,))
    for line_idx in range(num_lines):
        line_offset = -(-num_lines / 2 + line_idx) * line_spacing
        line_offsets[line_idx] = line_offset
        t_centers = np.sqrt(element_positions ** 2 + point_depth ** 2) / speed_of_sound
        t_delays = np.sqrt((element_positions - line_offset) ** 2 + point_depth ** 2) / speed_of_sound
        line_aperture_data[:, :, line_idx] = np.stack(
            [pulse(t_center - t_delay) for t_center, t_delay in zip(t_centers, t_delays)])

    # pulse_side = line_aperture_data[:, :, pulse_side_line_idx]
    pulse_side_distance = line_offsets[pulse_side_line_idx]

    image_tick_labels = line_offsets[127:0:-21]
    image_tick_labels = [f'${x:.1f}$' if idx % 2 == 0 else None for idx, x in enumerate(image_tick_labels)]
    image_ticks = [list(range(0, 127, 21)), image_tick_labels]

    # pulse_clutter = 0.5 * pulse_focus + 0.5 * pulse_side
    params = {'pulse_frequency': pulse_frequency, 'sampling_frequency': sampling_frequency,
              'line_offsets': line_offsets, 'pulse_side_distance': pulse_side_distance, 'image_ticks': image_ticks}

    return {'pulse_focus': pulse_focus, 'pulse_side': pulse_side, 'pulse_channel_data': pulse_channel_data,
            'line_aperture_data': line_aperture_data, 'params': params}