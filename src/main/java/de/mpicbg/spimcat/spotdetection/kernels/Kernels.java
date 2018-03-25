package de.mpicbg.spimcat.spotdetection.kernels;

import clearcl.ClearCLImage;
import clearcl.imagej.ClearCLIJ;
import de.mpicbg.spimcat.spotdetection.GPUSpotDetection;

import java.util.HashMap;

public class Kernels {
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

    public static boolean downsample(ClearCLIJ clij, ClearCLImage src, ClearCLImage dst, float factorX, float factorY, float factorZ) {
        HashMap<String, Object> parameters = new HashMap<>();
        parameters.put("src", src);
        parameters.put("dst", dst);
        parameters.put("factor_x", 1.f / factorX);
        parameters.put("factor_y", 1.f / factorY);
        parameters.put("factor_z", 1.f / factorZ);
        return clij.execute(Kernels.class, "downsampling.cl", "downsample_3d_nearest", parameters);
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

    public static boolean dilate(ClearCLIJ clij, ClearCLImage src, ClearCLImage dst) {
        HashMap<String, Object> parameters = new HashMap<>();
        parameters.put("src", src);
        parameters.put("dst", dst);
        return clij.execute(Kernels.class, "binaryProcessing.cl", "dilate_6_neighborhood_3d", parameters);
    }


    public static boolean erode(ClearCLIJ clij, ClearCLImage src, ClearCLImage dst) {
        HashMap<String, Object> parameters = new HashMap<>();
        parameters.put("src", src);
        parameters.put("dst", dst);
        return clij.execute(Kernels.class, "binaryProcessing.cl", "erode_6_neighborhood_3d", parameters);
    }
}
