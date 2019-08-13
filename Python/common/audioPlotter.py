import matplotlib.pyplot as plt
import numpy as np
import common.audioHandler as audioHandler

class AudioPlotter:

    def __init__(self):
        self.audio = audioHandler.AudioHandler()
        self.fig = plt.figure(f"Name...")
        plt.ion()

    # ############################################################################
    # Plots audio and spectrum for two channels. Any value may be None..
    # ############################################################################
    def plotSoundSimple(self, audio1, audio2=None, audio3=None, audio4=None, useSame=False, blocking=False):

        try:

            self.fig.clf()

            #if not blocking:
            #plt.close()

            plt.show()

            if audio1 is not None:
                # Add the L audio sound preassure graph
                if useSame:
                    axAudio = self.fig.add_subplot(111)
                else:
                    axAudio = self.fig.add_subplot(221)
                axAudio.plot(audio1['scaledData'], color='R')
                axAudio.set_xlabel(audio1["fileName"] + "" + audio1["userDefinedName"])
                axAudio.set_ylabel('Amplitude');

            if audio2 is not None:
                # Add the R audio sound preassure graph
                if useSame:
                    axAudio = self.fig.add_subplot(111)
                else:
                    axAudio = self.fig.add_subplot(222)
                axAudio.plot(audio2['scaledData'], color='G')
                axAudio.set_xlabel(audio2["fileName"] + "" + audio2["userDefinedName"])
                axAudio.set_ylabel('');

            if audio3 is not None:
                # Add the R audio sound preassure graph
                if useSame:
                    axAudio = self.fig.add_subplot(111)
                else:
                    axAudio = self.fig.add_subplot(223)
                axAudio.plot(audio3['scaledData'], color='B')
                axAudio.set_xlabel(audio3["fileName"] + "" + audio3["userDefinedName"])
                axAudio.set_ylabel('');

            if audio4 is not None:
                # Add the R audio sound preassure graph
                if useSame:
                    axAudio = self.fig.add_subplot(111)
                else:
                    axAudio = self.fig.add_subplot(224)
                axAudio.plot(audio4['scaledData'], color='Y')
                axAudio.set_xlabel(audio4["fileName"] + "" + audio4["userDefinedName"])
                axAudio.set_ylabel('');

            # Show!
            if not blocking:
                plt.draw()
            else:
                plt.show(block=True)
                plt.close()

            plt.pause(0.01)
        except Exception as ex:
            print(f"Plotting problems: {ex}")

    # ############################################################################
    # Plots spectrum, either from a provided sound, or a raw np array
    # ############################################################################
    def plotSoundAndSpectrum(self, audioL, audioR, spectrumListL, spectrumListR, rate, sampleCount, audioLengthS, name=None):


        print("OBSOLETE!!??")
        return

        nameField = " "
        if name != None:
            nameField = name + ", "

        f = plt.figure(f"{nameField} Rate: {rate/1000:.1f} kHz, duration: {audioLengthS:.2f} sec, samples/channel: {sampleCount}")

        if audioL is not None:
            # Add the L audio sound preassure graph
            axAudio = f.add_subplot(221)
            axAudio.plot(np.arange(sampleCount) / rate, audioL)
            axAudio.set_xlabel('Time (L)[s]')
            axAudio.set_ylabel('Amplitude');

        if audioR is not None:
            # Add the R audio sound preassure graph
            axAudio = f.add_subplot(222)
            axAudio.plot(np.arange(sampleCount) / rate, audioR)
            axAudio.set_xlabel('Time (R)[s]')
            axAudio.set_ylabel('Amplitude');

        if spectrumListL is not None:
            # Add the spectrum for Left channel
            axSpec = f.add_subplot(223)
            S = np.abs(spectrumListL)
            S = 20 * np.log10(S / np.max(S))

            axSpec.imshow(S, origin='lower', cmap='viridis',
                      extent=(0, audioLengthS, 0, rate / 2 / 1000))
            axSpec.axis('tight')
            axSpec.set_ylabel('Freq (L)[kHz]')
            axSpec.set_xlabel('Time (L)[s]');

        if spectrumListR is not None:
            # Add the spectrum for Left channel
            axSpec = f.add_subplot(224)
            S = np.abs(spectrumListR)
            S = 20 * np.log10(S / np.max(S))

            axSpec.imshow(S, origin='lower', cmap='viridis',
                      extent=(0, audioLengthS, 0, rate / 2 / 1000))
            axSpec.axis('tight')
            axSpec.set_ylabel('Freq (R)[kHz]')
            axSpec.set_xlabel('Time (R)[s]');

        # Show!
        plt.show()


    # ############################################################################
    # Plots audio and spectrum for two channels. Any value may be None..
    def plotSpectrum(self, audioLabel=None, rawOutput=None):

        assert (audioLabel is not None or rawOutput is not None)

        spectrumOutput = None

        # SORRY!
        if audioLabel is not None:
            if audioLabel["frequencySpectrum"] is None:
                self.audio.addFFTToSound(audioLabel)
            spectrum1 = audioLabel["frequencySpectrum"]
            freqArray1 = audioLabel["freqArray"]

        if rawOutput is not None:
            spectrumOutput = rawOutput["frequencySpectrum"]
            freqArrayOutput = rawOutput["freqArray"]


        f = plt.figure(f"Name...")
        plt.xlabel('(Blue=Label, Red=InferenceOut) Freq (Khz)')
        plt.ylabel('Power (dB) / 100')


        if spectrum1 is not None:
            axAudio = f.add_subplot(111)

            # Plot the frequency
            axAudio.plot(freqArray1 / 1000, spectrum1, linewidth=1, color='B')

        if spectrumOutput is not None:
            axAudio = f.add_subplot(111)

            # Plot the frequency
            axAudio.plot(freqArrayOutput / 1000, spectrumOutput, linewidth=1, color='R')


        plt.show()
    # ############################################################################
