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
import ij.plugin.RGBStackMerge;

import java.io.IOException;
import java.util.HashMap;

public class GPUSpotDetection {
    private static float samplingFactorX = 0.5f;
    private static float samplingFactorY = 0.5f;
    private static float samplingFactorZ = 1;
    private static float threshold = 400;

    private static boolean cropPartForDebugging = true;

    private static ClearCLIJ clij;
    private static ImagePlus imp;

    public static void main (String... args) throws IOException {
        new ImageJ();
        clij = new ClearCLIJ("HD");
        ElapsedTime.sStandardOutput = true;

        String file = "";
        if (System.getProperty("os.name").startsWith("Windows")) {
            file = "C:\\structure\\data\\Uncalibrated.tif";
        } else {
            file = "/home/rhaase/data/Uncalibrated.tif";
        }

        imp = IJ.openImage(file);
        imp.show();

        for (int i = 0; i < 10; i++) {
            ElapsedTime.measure("the whole thing ", () -> {
                exec();
            });
        }
    }

    private static void exec() {

        ClearCLImage clImp = clij.converter(imp).getClearCLImage();
        ClearCLImage clInput = clImp;
        long[] targetDimensions;
        if (cropPartForDebugging) {
            targetDimensions = Arrays.elementwiseMultiplVectors(clImp.getDimensions(), new float[]{0.5f, 0.5f, 1});
            clInput = clij.createCLImage(targetDimensions, clImp.getChannelDataType());
            Kernels.crop(clij, clImp, clInput, 256, 0, 0);
        }

        targetDimensions = Arrays.elementwiseMultiplVectors(clInput.getDimensions(), new float[]{samplingFactorX, samplingFactorY, samplingFactorZ});
        ClearCLImage flip = clij.createCLImage(targetDimensions, clInput.getChannelDataType());
        ClearCLImage flop = clij.createCLImage(targetDimensions, clInput.getChannelDataType());
        ClearCLImage flap = clij.createCLImage(targetDimensions, clInput.getChannelDataType());

        Kernels.downsample(clij, clInput, flop, samplingFactorX, samplingFactorY, samplingFactorZ);
        clij.show(flop, "downsampled");
        ImagePlus downsampledImp = IJ.getImage();


        Kernels.blurSlicewise(clij, flop, flip, 6, 6, 3f, 3f);

        Kernels.threshold(clij, flip, flap, threshold);

        Kernels.addScalar(clij, flop, flip, -threshold);
        Kernels.mask(clij, flip, flap, flop);

        //clij.show(flop, "src_dog");



        // Spot detection
        Kernels.differenceOfGaussian(clij, flop, flip, 6, 1.1f, 5f);
        //clij.show(flip, "dog");

        Kernels.detectMaxima(clij, flip, flop, 3);
        //clij.show(flop, "detected maxima");

        System.out.println("Count: " + new GPUSum(clij, flop).sum());

        Kernels.dilate(clij, flop, flip);
        Kernels.dilate(clij, flip, flop);
        //Kernels.dilate(clij, flop, flip);
        clij.show(flop, "2x dilated");

        ImagePlus spotsImp = IJ.getImage();

        ImagePlus merged = RGBStackMerge.mergeChannels(new ImagePlus[]{downsampledImp, spotsImp}, true);
        merged.show();

        System.out.print("Bye");
    }
}