package de.mpicbg.spimcat.spotdetection;

import clearcl.ClearCLImage;
import clearcl.imagej.ClearCLIJ;
import clearcl.util.ElapsedTime;
import clearcl.imagej.kernels.Kernels;
import de.mpicbg.spimcat.spotdetection.math.Arrays;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.plugin.RGBStackMerge;

import java.io.IOException;

public class GPUSpotDetection {
    private float samplingFactorX = 0.5f;
    private float samplingFactorY = 0.5f;
    private float samplingFactorZ = 1;
    private static float threshold = 400;

    private static boolean cropPartForDebugging = true;

    private ClearCLIJ clij;
    //private static ImagePlus imp;

    private ClearCLImage input;
    private ClearCLImage output;

    public GPUSpotDetection(ClearCLIJ clij, ClearCLImage input, ClearCLImage output, float threshold) {
      this.clij = clij;
      this.output = output;
      this.input = input;
      this.threshold = threshold;

      samplingFactorX = (float)output.getWidth() / input.getWidth();
      samplingFactorY = (float)output.getHeight() / input.getHeight();
      samplingFactorZ = (float)output.getDepth() / input.getDepth();


    }

    public void exec() {

        //ClearCLImage clImp = clij.converter(imp).getClearCLImage();
        //ClearCLImage input = clImp;
        long[] targetDimensions = new long[]{output.getWidth(), output.getHeight(), output.getDepth()};
        /*if (cropPartForDebugging) {
            targetDimensions = Arrays.elementwiseMultiplVectors(clImp.getDimensions(), new float[]{0.5f, 0.5f, 1});
            clInput = clij.createCLImage(targetDimensions, clImp.getChannelDataType());
            Kernels.crop(clij, clImp, clInput, 256, 0, 0);
        }*/

        //targetDimensions = Arrays.elementwiseMultiplVectors(clInput.getDimensions(), new float[]{samplingFactorX, samplingFactorY, samplingFactorZ});
        ClearCLImage flip = clij.createCLImage(targetDimensions, input.getChannelDataType());
        ClearCLImage flop = output; //clij.createCLImage(targetDimensions, clInput.getChannelDataType());
        ClearCLImage flap = clij.createCLImage(targetDimensions, input.getChannelDataType());

        System.out.println("factors " + samplingFactorX + "/" + samplingFactorY + "/" + samplingFactorZ);
        Kernels.downsample(clij, input, flop, samplingFactorX, samplingFactorY, samplingFactorZ);
        clij.show(flop, "downsampled");

        ImagePlus downsampledImp = IJ.getImage();


        Kernels.blurSlicewise(clij, flop, flip, 6, 6, 3f, 3f);

        Kernels.threshold(clij, flip, flap, threshold);

        Kernels.addScalar(clij, flop, flip, -threshold);
        Kernels.mask(clij, flip, flap, flop);

        //clij.show(flop, "src_dog");



        // Spot detection
        Kernels.differenceOfGaussian(clij, flop, flip, 6, 1.5f, 3f);
        clij.show(flip, "dog");
        ImagePlus dogImp = IJ.getImage();

        Kernels.detectMaxima(clij, flip, flop, 3);
        //clij.show(flop, "detected maxima");

        System.out.println("Count: " + Kernels.sumPixels(clij, flop));

        Kernels.dilate(clij, flop, flip);
        Kernels.dilate(clij, flip, flop);
        //Kernels.dilate(clij, flop, flip);
        clij.show(flop, "2x dilated");

        ImagePlus spotsImp = IJ.getImage();

        ImagePlus merged = RGBStackMerge.mergeChannels(new ImagePlus[]{downsampledImp, spotsImp, dogImp}, true);
        merged.show();

        flip.close();
        flap.close();
        System.out.print("Bye");
    }
}