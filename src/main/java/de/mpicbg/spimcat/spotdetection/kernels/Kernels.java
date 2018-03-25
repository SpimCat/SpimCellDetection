package de.mpicbg.spimcat.spotdetection.kernels;

import clearcl.ClearCLImage;
import clearcl.imagej.ClearCLIJ;
import clearcl.imagej.demo.BenchmarkingDemo;
import de.mpicbg.spimcat.spotdetection.GPUSpotDetection;
import fastfuse.tasks.GaussianBlurTask;

import java.io.IOException;
import java.util.HashMap;

public class Kernels {

    public static boolean addPixelwise(ClearCLIJ pCLIJ, ClearCLImage src, ClearCLImage src1, ClearCLImage dst) {
        HashMap<String, Object> lParameters = new HashMap<>();
        lParameters.put("src", src);
        lParameters.put("src1", src1);
        lParameters.put("dst", dst);
        return pCLIJ.execute(Kernels.class, "math.cl", "addPixelwise", lParameters);
    }


    public static boolean addScalar(ClearCLIJ pCLIJ, ClearCLImage src, ClearCLImage dst, float scalar) {
        HashMap<String, Object> lParameters = new HashMap<>();
        lParameters.put("src", src);
        lParameters.put("scalar", scalar);
        lParameters.put("dst", dst);
        return pCLIJ.execute(Kernels.class, "math.cl", "addScalar", lParameters);
    }

    public static boolean addWeightedPixelwise(ClearCLIJ pCLIJ, ClearCLImage src, ClearCLImage src1, ClearCLImage dst, float factor, float factor1) {
        HashMap<String, Object> lParameters = new HashMap<>();
        lParameters.put("src", src);
        lParameters.put("src1", src1);
        lParameters.put("factor", factor);
        lParameters.put("factor1", factor1);
        lParameters.put("dst", dst);
        return pCLIJ.execute(Kernels.class, "math.cl", "addWeightedPixelwise", lParameters);
    }

    public static boolean blur(ClearCLIJ pCLIJ, ClearCLImage src, ClearCLImage dst, int nX, int nY, int nZ, float sigmaX, float sigmaY, float sigmaZ){
        HashMap<String, Object> lParameters = new HashMap<>();
        lParameters.put("Nx", nX);
        lParameters.put("Ny", nY);
        lParameters.put("Nz", nZ);
        lParameters.put("sx", sigmaX);
        lParameters.put("sy", sigmaY);
        lParameters.put("sz", sigmaZ);
        lParameters.put("src", src);
        lParameters.put("dst", dst);
        return pCLIJ.execute(Kernels.class, "blur.cl", "gaussian_blur_image3d", lParameters);
    }

    public static boolean blurSlicewise(ClearCLIJ pCLIJ, ClearCLImage src, ClearCLImage dst, int nX, int nY, float sigmaX, float sigmaY){
        HashMap<String, Object> lParameters = new HashMap<>();
        lParameters.put("Nx", nX);
        lParameters.put("Ny", nY);
        lParameters.put("sx", sigmaX);
        lParameters.put("sy", sigmaY);
        lParameters.put("src", src);
        lParameters.put("dst", dst);
        return pCLIJ.execute(Kernels.class, "blur.cl", "gaussian_blur_slicewise_image3d", lParameters);
    }


    public static boolean copy(ClearCLIJ clij, ClearCLImage src, ClearCLImage dst) {
        HashMap<String, Object> parameters = new HashMap<>();
        parameters.put("src", src);
        parameters.put("dst", dst);
        return clij.execute(Kernels.class, "duplication.cl", "copy", parameters);
    }

    public static boolean crop(ClearCLIJ clij, ClearCLImage src, ClearCLImage dst, int startX, int startY, int startZ) {
        HashMap<String, Object> parameters = new HashMap<>();
        parameters.put("src", src);
        parameters.put("dst", dst);
        parameters.put("start_x", startX);
        parameters.put("start_y", startY);
        parameters.put("start_z", startZ);
        return clij.execute(Kernels.class, "duplication.cl", "crop", parameters);
    }

    public static boolean detectMaxima(ClearCLIJ clij, ClearCLImage src, ClearCLImage dst, int radius) {
        return detectOptima(clij, src, dst, radius, true);
    }

    public static boolean detectMinima(ClearCLIJ clij, ClearCLImage src, ClearCLImage dst, int radius) {
        return detectOptima(clij, src, dst, radius,  false);
    }

    public static boolean detectOptima(ClearCLIJ clij, ClearCLImage src, ClearCLImage dst, int radius, boolean detectMaxima) {
        HashMap<String, Object> parameters = new HashMap<>();
        parameters.put("src", src);
        parameters.put("dst", dst);
        parameters.put("radius", radius);
        parameters.put("detect_maxima", detectMaxima?1:0);
        return clij.execute(Kernels.class, "detection.cl", "detect_local_optima_3d", parameters);
    }

    public static boolean differenceOfGaussian(ClearCLIJ clij, ClearCLImage src, ClearCLImage dst, int radius, float sigmaMinuend, float sigmaSubtrahend) {
        HashMap<String, Object> parameters = new HashMap<>();
        parameters.put("src", src);
        parameters.put("dst", dst);
        parameters.put("radius", radius);
        parameters.put("sigma_minuend", sigmaMinuend);
        parameters.put("sigma_subtrahend", sigmaSubtrahend);
        return clij.execute(Kernels.class, "differenceOfGaussian.cl", "subtract_convolved_images_3d_fast", parameters);
    }

    public static boolean dilate(ClearCLIJ clij, ClearCLImage src, ClearCLImage dst) {
        HashMap<String, Object> parameters = new HashMap<>();
        parameters.put("src", src);
        parameters.put("dst", dst);
        return clij.execute(Kernels.class, "binaryProcessing.cl", "dilate_6_neighborhood_3d", parameters);
    }

    public static boolean downsample(ClearCLIJ clij, ClearCLImage src, ClearCLImage dst, float factorX, float factorY, float factorZ) {
        HashMap<String, Object> parameters = new HashMap<>();
        parameters.put("src", src);
        parameters.put("dst", dst);
        parameters.put("factor_x", 1.f / factorX);
        parameters.put("factor_y", 1.f / factorY);
        parameters.put("factor_z", 1.f / factorZ);
        return clij.execute(Kernels.class, "downsampling.cl", "downsample_3d_nearest", parameters);
    }

    public static boolean erode(ClearCLIJ clij, ClearCLImage src, ClearCLImage dst) {
        HashMap<String, Object> parameters = new HashMap<>();
        parameters.put("src", src);
        parameters.put("dst", dst);
        return clij.execute(Kernels.class, "binaryProcessing.cl", "erode_6_neighborhood_3d", parameters);
    }

    public static boolean mask(ClearCLIJ pCLIJ, ClearCLImage src, ClearCLImage mask, ClearCLImage dst) {
        HashMap<String, Object> lParameters = new HashMap<>();
        lParameters.put("src", src);
        lParameters.put("mask", mask);
        lParameters.put("dst", dst);

        return pCLIJ.execute(Kernels.class, "mask.cl", "mask", lParameters);
    }

    public static boolean multiplyPixelwise(ClearCLIJ pCLIJ, ClearCLImage pImage, ClearCLImage pImage1, ClearCLImage pOutputImage) {
        HashMap<String, Object> lParameters = new HashMap<>();
        lParameters.put("src", pImage);
        lParameters.put("src1", pImage1);
        lParameters.put("dst", pOutputImage);
        return pCLIJ.execute(Kernels.class, "math.cl", "multiplyPixelwise", lParameters);
    }

    public static boolean threshold(ClearCLIJ clij, ClearCLImage src, ClearCLImage dst, float threshold) {
        HashMap<String, Object> lParameters = new HashMap<>();

        lParameters.clear();
        lParameters.put("threshold", threshold);
        lParameters.put("src", src);
        lParameters.put("dst", dst);
        return clij.execute(Kernels.class, "thresholding.cl", "applyThreshold", lParameters);
    }

}
