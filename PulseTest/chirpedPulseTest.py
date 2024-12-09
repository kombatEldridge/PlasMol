import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import re

def testParams(t=np.linspace(0, 20, 1000),
               frequency=1, 
               peakTime=10, 
               width=0.2):
    frequency = [frequency] if not isinstance(frequency, list) else frequency
    peakTime = [peakTime] if not isinstance(peakTime, list) else peakTime
    width = [width] if not isinstance(width, list) else width

    plt.figure(figsize=(10, 6))
    for freq in frequency:
        for peak in peakTime:
            for wid in width:
                nonchirp = np.exp(1j * 2 * np.pi * freq * (t - peak)) * np.exp(-wid * (t - peak)**2)
                plt.plot(t, np.real(nonchirp), label=f"Freq={freq}, Peak={peak}, Width={wid}")

    plt.legend()
    plt.title("Pulse")
    plt.show()


def getWavelength(filePath):

    def regex(file_name):
        frequency_match = re.search(r"f(\d+)", file_name)
        width_match = re.search(r"w([\d.]+)", file_name)
        peak_time_match = re.search(r"pT(\d+)", file_name)

        frequency = int(frequency_match.group(1)) if frequency_match else None
        width = float(width_match.group(1)) if width_match else None
        peak_time = int(peak_time_match.group(1)) if peak_time_match else None

        return frequency, width, peak_time

    def pulse_wave(t, frequency, peak_time, width, a):
        return a * np.exp(1j * 2 * np.pi * frequency * (t - peak_time)) * np.exp(-width * (t - peak_time)**2)

    def real_part(t, frequency, peak_time, width, a):
        return np.real(pulse_wave(t, frequency, peak_time, width, a))
    
    def final(t, conv_factor):
        frequency = initFreq * (1/conv_factor)
        width = initWidth * (1/(conv_factor**2))
        peak_time = initPeak * conv_factor
        a = amp
        return real_part(t, frequency, peak_time, width, a)

    data = pd.read_csv(filePath, comment='#') 
    time = data['Timestamps (fs)'] 
    intensity = data['Z Values'] 

    reg = regex(filePath)
    initFreq = reg[0]
    initWidth = reg[1]
    initPeak = reg[2]
    amp = max(intensity)

    initial_guess = 3.365584237

    param, covariance = curve_fit(final, time, intensity, p0=initial_guess, maxfev = 1000)
    conv_fac = param[0]

    # plt.figure(figsize=(10, 6))
    # plt.plot(time, intensity, label="Data", color='black')
    # plt.plot(time, final(time, conv_fac), label="Fitted Wave", color='red', linestyle='--')
    # plt.xlabel("Time (fs)")
    # plt.ylabel("Intensity")
    # plt.legend()
    # plt.show()

    print(f"Filename: {filePath}")
    print(f"Fitted Conversion Factor: {conv_fac}\n")

if __name__ == '__main__':
    getWavelength('pulse_f0.5_w0.2_pT10_r1000_tT50-E-Field.csv')
    getWavelength('pulse_f1_w0.2_pT8_r1000_tT50-E-Field.csv')
    getWavelength('pulse_f1_w0.2_pT10_r1000_tT50-E-Field.csv')
    getWavelength('pulse_f1_w0.2_pT12_r1000_tT50-E-Field.csv')
    getWavelength('pulse_f1_w0.3_pT10_r1000_tT50-E-Field.csv')
    getWavelength('pulse_f1_w0.4_pT10_r1000_tT50-E-Field.csv')
    getWavelength('pulse_f2_w0.2_pT10_r1000_tT50-E-Field.csv')
    getWavelength('pulse_f3_w0.2_pT10_r1000_tT50-E-Field.csv')