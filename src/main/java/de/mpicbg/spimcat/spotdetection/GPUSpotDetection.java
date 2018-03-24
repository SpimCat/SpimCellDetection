package de.mpicbg.spimcat.spotdetection;

import clearcl.ClearCLImage;
import clearcl.imagej.ClearCLIJ;
import clearcl.imagej.demo.BenchmarkingDemo;
import clearcl.util.ElapsedTime;
import de.mpicbg.spimcat.spotdetection.math.Arrays;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;

import java.io.IOException;
import java.util.HashMap;

public class GPUSpotDetection {
    public static void main (String... args) throws IOException {

        float samplingFactorX = 0.25f;
        float samplingFactorY = 0.25f;
        float samplingFactorZ = 1;

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
        parameters.put("src", clInput);
        parameters.put("dst", flop);
        parameters.put("factor_x", 1.f / samplingFactorX);
        parameters.put("factor_y", 1.f / samplingFactorY);
        parameters.put("factor_z", 1.f / samplingFactorZ);
        clij.execute(GPUSpotDetection.class, "kernels/downsampling.cl", "downsample_3d_nearest", parameters);
        clij.converter(flop).getImagePlus().show();

        parameters.clear();
        parameters.put("input", flop);
        parameters.put("output", flip);
        parameters.put("radius",6);
        parameters.put("sigma_minuend",1.5f);
        parameters.put("sigma_subtrahend",3f);
        clij.execute(GPUSpotDetection.class, "kernels/differenceOfGaussian.cl", "subtract_convolved_images_3d_fast", parameters);
        clij.converter(flip).getImagePlus().show();

        parameters.clear();
        parameters.put("input", flip);
        parameters.put("output", flop);
        parameters.put("radius", 3);
        parameters.put("detect_maxima", 1);
        clij.execute(GPUSpotDetection.class, "kernels/detection.cl", "detect_local_optima_3d", parameters);
        clij.show(flop, "detected maxima");

        parameters.clear();
        parameters.put("src", flop);
        parameters.put("dst", flip);
        clij.execute(GPUSpotDetection.class, "kernels/binaryProcessing.cl", "dilate_6_neighborhood_3d", parameters);
        parameters.clear();
        parameters.put("src", flip);
        parameters.put("dst", flop);
        clij.execute(GPUSpotDetection.class, "kernels/binaryProcessing.cl", "dilate_6_neighborhood_3d", parameters);
        parameters.clear();
        parameters.put("src", flop);
        parameters.put("dst", flip);
        clij.execute(GPUSpotDetection.class, "kernels/binaryProcessing.cl", "dilate_6_neighborhood_3d", parameters);
        clij.show(flip, "3x dilated");




        System.out.print("Bye");
    }
}
