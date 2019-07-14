package Jorge.Pineda.verifier;

import android.util.Log;
import org.jtransforms.fft.*;

//java translation of speechpy implementation

public class Processing {

    double[][] stack_frames(double[] signal, int sampling_frequency, double frame_length, double frame_stride) {
        // Initial values
        int length_signal = signal.length;
        int frame_sample_length = (int) Math.round(sampling_frequency * frame_length);
        frame_stride = Math.round(sampling_frequency * frame_stride);

        //Zero padding
        int numframes = (int) Math.ceil((length_signal - frame_sample_length) / frame_stride);
        int len_sig = (int) (numframes * frame_stride + frame_sample_length);
        double padded_signal[] = new double[len_sig];
        System.arraycopy(signal, 0, padded_signal, 0, length_signal);

        //Create frames array
        double frames[][] = new double[numframes][frame_sample_length];

        //Assign frames
        int added_stride = 0;
        for (int i=0; i<numframes; i++) {
            for (int j=0; j<frame_sample_length; j++) {
                frames[i][j] = padded_signal[j + added_stride];
            }
            added_stride += frame_stride;
        }

        return frames;

    }

    double[][] power_spectrum(double[][] frames, int fft_points) {
        DoubleFFT_1D fft = new DoubleFFT_1D(fft_points);
        int numframes = frames.length;
        int framelen = (fft_points/2) + 1;
        double result[][] = new double[numframes][framelen];

        //Calculate fft of each frame and square each element
        for (int i=0; i<numframes-1; i++) {
            double a[] = new double[fft_points];
            System.arraycopy(frames[i], 0, a, 0, fft_points);
            fft.realForward(a);
            for (int j=1; j<framelen-1; j++) {
                double temp = Math.sqrt(a[2*j]*a[2*j] + a[2*j+1]*a[2*j+1]);
                result[i][j]= temp * temp * (1/ (double) fft_points);
            }
            result[i][0] = a[0] * a[0] * (1/(double) fft_points);
            result[i][framelen - 1] = a[1] * a[1] * (1/(double) fft_points);

        }
        return result;

    }

    double convert_to_mel(double frequency) {
        return 2595 * Math.log10(1 + (frequency / 700));

    }

    double convert_to_freq(double mel) {
        return 700 * (Math.pow(10, mel / 2595) - 1);

    }

    double[] linspace(double start, double stop, int num) {
        double space = (stop - start) / num;
        double result[] = new double[num];
        double counter = start;
        for (int i = 0; i<num; i++) {
            result[i] = counter;
            counter += space;
        }
        return result;

    }

    double[] triangle(double[] x, int left, int middle, int right) {
        double[] out = new double[x.length];
        for (int i = 0; i<x.length; i++) {
            if (left < x[i] && x[i] <= middle) {
                out[i] = (x[i] - left) / (middle - left);
            }
        }
        for (int i = 0; i<x.length; i++) {
            if (middle <= x[i] && x[i] < right) {
                out[i] = (right - x[i]) / (right - middle);
            }
        }
        return out;

    }

    double[][] filterbanks(int num_filter, int coefficients, int sampling_freq, double low_freq, double high_freq) {
        //Convert frequencies to mel
        double high_mel = convert_to_mel(high_freq);
        double low_mel = convert_to_mel(300);

        //Computing mel filterbanks
         double mels[] = linspace(low_mel, high_mel, num_filter + 2);

         //Convert mel to hertz
        int mels_len = mels.length;
        double hertz[] = new double[mels_len];
        for (int i = 0; i<mels_len; i++) {
            hertz[i] = convert_to_freq(mels[i]);
        }

        //Round frequencies to the closest FFT bin
        double freq_index[] = new double[mels_len];
        for (int i = 0; i< mels_len; i++) {
            freq_index[i] = Math.floor((coefficients + 1) * hertz[i] / sampling_freq);
        }

        //Initial definition
        double[][] filterbank = new double[num_filter][coefficients];

        //The triangular function for each filter
        for (int i = 0; i< num_filter; i++) {
            int left = (int) freq_index[i];
            int middle = (int) freq_index[i + 1];
            int right = (int) freq_index[i + 2];
            double[] z = linspace(left, right, right - left + 1);
            double [] temp = triangle(z, left, middle, right);
            for (int j = left; j <= right; j++) {
                filterbank[i][j] = temp[j - left];
            }

        }

        return filterbank;

    }

    double[][] mfe(double[] signal, int sampling_frequency, double frame_length, double frame_stride, int num_filters , int fft_length) {

        //Stack frames
        double frames[][] = stack_frames(signal,sampling_frequency,frame_length,frame_stride);

        //Getting high frequency
        double high_frequency =((double) sampling_frequency) / 2;

        //Calculate power spectrum
        double power_spectrum[][] = power_spectrum(frames, fft_length);
        int coefficients = power_spectrum[0].length;

        //Extracting the filterbanks
        double filter_banks[][] = filterbanks(num_filters, coefficients, sampling_frequency, 0, high_frequency);
        //Filterbank energies
        double features[][] = new double[power_spectrum.length][num_filters];
        for (int i =0; i<power_spectrum.length; i++) {
            for (int j =0; j<num_filters; j++) {
                for (int k = 0; k<coefficients; k++) {
                    features[i][j] += power_spectrum[i][k] * filter_banks[j][k];
                }
            }
        }

        return features;

    }

    float[][] lmfe(double[] signal, int sampling_frequency, double frame_length, double frame_stride, int num_filters, int fft_length) {
        double features[][] = mfe(signal, sampling_frequency, frame_length, frame_stride, num_filters, fft_length);
        float featuresf[][] = new float[features.length][features[0].length];
        for (int i = 0; i<features.length; i++) {
            for (int j = 0; j<features[0].length; j++) {
                //Log.d("DEBUG","hole 1 " + features[i][j]);
                if (features[i][j]!=0) {
                    featuresf[i][j] = (float) Math.log(features[i][j]);
                }
                //Log.d("DEBUG","hole 2 " + featuresf[i][j]);
            }

        }
        return featuresf;

    }

    float[][] derivative_extraction(float[][] features, int DeltaWindows) {
        //Getting shape of the vector
        int rows = features.length;
        int cols = features[0].length;

        //Defining the vector of differences
        float DIF[][] = new float[rows][cols];
        int Scale = 0;

        //Pad only along features in the vector
        float FEAT[][] = new float[rows][cols + (2 * DeltaWindows)];
        for (int i = 0; i< rows; i++) {
            System.arraycopy(features[i], 0, FEAT[i], DeltaWindows, cols);
            float temp = features[i][0];
            for (int j = 0; j < DeltaWindows; j++) {
                FEAT[i][j] = temp;
            }
            temp = features[i][cols - 1];
            for (int j = DeltaWindows + cols; j < cols + (2 * DeltaWindows); j++) {
                FEAT[i][j] = temp;
            }
        }

        //Derivative extraction
        for (int i = 0; i < DeltaWindows; i++) {
            int offset = DeltaWindows;
            int Range = i + 1;
            for (int j = 0; j < rows; j++) {
                for (int k = 0; k < cols; k++) {
                    DIF[j][k] += Range * FEAT[j][offset + Range + k] /*- FEAT[j][offset - Range + k]*/;
                }
            }
            Scale += 2 * Math.pow(Range, 2);

        }
        for (int j = 0; j < rows; j++) {
            for (int k = 0; k < cols; k++) {
                DIF[j][k] = DIF[j][k] / Scale;
            }
        }

        return DIF;


    }

    float[][][] extract_derivative_features(float[][] features) {
        float feature_cube[][][] = new float[/*features.length*/100][features[0].length][3];
        float features1[][] = derivative_extraction(features, 2);
        float features2[][] = derivative_extraction(features1, 2);

        for(int i = 0; i< feature_cube.length ; i++) {
            for(int j = 0; j < feature_cube[0].length; j++) {
                feature_cube[i][j][0] = features[i+30][j];
                feature_cube[i][j][1] = features1[i+30][j];
                feature_cube[i][j][2] = features2[i+30][j];

            }

        }

        return feature_cube;
    }


}
