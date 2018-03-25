package de.mpicbg.spimcat.spotdetection;

import clearcl.ClearCLImage;
import clearcl.imagej.ClearCLIJ;
import clearcl.imagej.demo.BenchmarkingDemo;
import clearcl.util.ElapsedTime;
import de.mpicbg.spimcat.spotdetection.kernels.Kernels;
import de.mpicbg.spimcat.spotdetection.math.Arrays;
import de.mpicbg.spimcat.spotdetection.tools.GPUSum;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;

import java.io.IOException;
import java.util.HashMap;

public class GPUSpotDetection {
    public static void main (String... args) throws IOException {

        float samplingFactorX = 0.5f;
        float samplingFactorY = 0.5f;
        float samplingFactorZ = 2;

        new ImageJ();
        ClearCLIJ clij = new ClearCLIJ("HD");
        ElapsedTime.sStandardOutput = true;

        ImagePlus imp = IJ.openImage("C:\\structure\\data\\Uncalibrated.tif");
        imp.show();

        ClearCLImage clInput = clij.converter(imp).getClearCLImage();
        long[] targetDimensions = Arrays.elementwiseMultiplVectors(clInput.getDimensions(), new float[]{samplingFactorX, samplingFactorY, samplingFactorZ});
        ClearCLImage flip = clij.createCLImage(targetDimensions, clInput.getChannelDataType());
        ClearCLImage flop = clij.createCLImage(targetDimensions, clInput.getChannelDataType());

        HashMap<String, Object> parameters = new HashMap<>();
        Kernels.downsample(clij, clInput, flop, 0.5f, 0.5f, 1f);
        clij.show(flop, "downsampled");

        parameters.clear();
        Kernels.differenceOfGaussian(clij, flop, flip, 6, 1.5f, 3f);
        clij.show(flip, "dog");

        parameters.clear();
        Kernels.detectMaxima(clij, flip, flop, 3);
        clij.show(flop, "detected maxima");

        System.out.println("Count: " + new GPUSum(clij, flop).sum());

        Kernels.dilate(clij, flop, flip);
        Kernels.dilate(clij, flip, flop);
        Kernels.dilate(clij, flop, flip);
        clij.show(flip, "3x dilated");




        System.out.print("Bye");
    }
}
