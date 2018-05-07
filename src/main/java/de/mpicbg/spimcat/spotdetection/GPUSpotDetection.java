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

    private float threshold = 400;
    private float dogSigmaMinuend = 3f;
    private float dogSigmaSubtrahend = 6f;
    private int dogRadius = 3;
    private float blurSigma = 6f;
    private int blurRadius = 12;

    private int optimaDetectionRadius = 3;

    private boolean showIntermediateResults = true;

    private ClearCLIJ clij;

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
                         + samplingFactorY
                         + "/"
                         + samplingFactorZ);
      Kernels.downsample(clij,
                         input,
                         flop,
                         samplingFactorX,
                         samplingFactorY,
                         samplingFactorZ);
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
      Kernels.differenceOfGaussian(clij,
                                   flop,
                                   flip,
                                   dogRadius,
                                   dogSigmaMinuend,
                                   dogSigmaSubtrahend);
      if (showIntermediateResults)
      {
        clij.show(flip, "dog");
        dogImp = IJ.getImage();
      }
      Kernels.detectMaxima(clij, flip, flop, optimaDetectionRadius);

      System.out.println("Spot count: " + Kernels.sumPixels(clij, flop));

      // result visualisation
      if (showIntermediateResults)
      {
          Kernels.dilate(clij, flop, flip);
          Kernels.dilate(clij, flip, flap);

          clij.show(flap, "2x dilated");

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

  public void setThreshold(float threshold)
  {
    this.threshold = threshold;
  }

  public void setDogSigmaMinuend(float dogSigmaMinuend)
  {
    this.dogSigmaMinuend = dogSigmaMinuend;
  }

  public void setDogSigmaSubtrahend(float dogSigmaSubtrahend)
  {
    this.dogSigmaSubtrahend = dogSigmaSubtrahend;
  }

  public void setDogRadius(int dogRadius)
  {
    this.dogRadius = dogRadius;
  }

  public void setBlurSigma(float blurSigma)
  {
    this.blurSigma = blurSigma;
  }

  public void setBlurRadius(int blurRadius)
  {
    this.blurRadius = blurRadius;
  }

  public void setOptimaDetectionRadius(int optimaDetectionRadius)
  {
    this.optimaDetectionRadius = optimaDetectionRadius;
  }

  public void setShowIntermediateResults(boolean showIntermediateResults)
  {
    this.showIntermediateResults = showIntermediateResults;
  }
}