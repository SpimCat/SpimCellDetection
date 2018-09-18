package de.mpicbg.spimcat.spotdetection;

import clearcl.ClearCLImage;
import clearcl.imagej.ClearCLIJ;
import clearcl.imagej.kernels.Kernels;
import ij.IJ;
import ij.ImagePlus;
import ij.plugin.RGBStackMerge;

/**
 * GPUSpotDetectionSliceBySlice
 * <p>
 * <p>
 * <p>
 * Author: @haesleinhuepf
 * 09 2018
 */
public class GPUSpotDetectionSliceBySlice extends GPUSpotDetection {

    public GPUSpotDetectionSliceBySlice(ClearCLIJ clij, ClearCLImage input, ClearCLImage output, float threshold) {
        super(clij, input, output, threshold);
    }

    public void exec()
    {

        // memory allocation
        long[]
                targetDimensions =
                new long[] { output.getWidth(),
                        output.getHeight(),
                        output.getDepth() };
        ClearCLImage flip = clij.createCLImage(targetDimensions, input.getChannelDataType());
        ClearCLImage flop = output; //clij.createCLImage(targetDimensions, clInput.getChannelDataType());
        ClearCLImage flap = clij.createCLImage(targetDimensions, input.getChannelDataType());
        ImagePlus downsampledImp = null;
        ImagePlus dogImp = null;

        // downsampling
        System.out.println("factors "
                + samplingFactorX
                + "/"
                + samplingFactorY);
        Kernels.downsample(clij,
                input,
                flop,
                samplingFactorX,
                samplingFactorY,
                1.0f);
        if (showIntermediateResults)
        {
            clij.show(flop, "downsampled");
            downsampledImp = IJ.getImage();
        }

        // blur
        Kernels.blurSlicewise(clij,
                flop,
                flip,
                blurRadius,
                blurRadius,
                blurSigma,
                blurSigma);

        // threshold
        Kernels.threshold(clij, flip, flap, threshold);
        Kernels.addScalar(clij, flop, flip, -threshold);

        // mask downsampled image
        Kernels.mask(clij, flip, flap, flop);

        // Spot detection
        Kernels.differenceOfGaussianSliceBySlice(clij,
                flop,
                flip,
                dogRadius,
                dogSigmaMinuend,
                dogSigmaSubtrahend);
        if (showIntermediateResults)
        {
            clij.show(flip, "dog sbs");
            dogImp = IJ.getImage();
        }


        // remove pixels below relative threshold (e.g. negative pixels after DoG)
        Kernels.threshold(clij, flip, flap, relativeThreshold);
        Kernels.mask(clij, flip, flap, flop);
        Kernels.copy(clij, flop, flip);

        Kernels.detectMaximaSliceBySlice(clij, flip, flop, optimaDetectionRadius);

        System.out.println("Spot count: " + Kernels.sumPixels(clij, flop));

        // result visualisation
        if (showIntermediateResults)
        {
            clij.show(flap, "result");

            ImagePlus spotsImp = IJ.getImage();
            ImagePlus
                    merged =
                    RGBStackMerge.mergeChannels(new ImagePlus[] { downsampledImp,
                            spotsImp,
                            dogImp }, true);
            merged.show();
        }
        flip.close();
        flap.close();
        System.out.print("Bye");
    }
}
