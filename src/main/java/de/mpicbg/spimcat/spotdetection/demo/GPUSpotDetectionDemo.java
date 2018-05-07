package de.mpicbg.spimcat.spotdetection.demo;

import clearcl.ClearCLImage;
import clearcl.imagej.ClearCLIJ;
import clearcl.imagej.kernels.Kernels;
import clearcl.util.ElapsedTime;
import de.mpicbg.spimcat.spotdetection.GPUSpotDetection;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;

import java.io.IOException;

/**
 * Author: Robert Haase (http://haesleinhuepf.net) at MPI CBG (http://mpi-cbg.de)
 * April 2018
 */
public class GPUSpotDetectionDemo
{

  public static void main (String... args) throws IOException
  {
    final float threshold = 120;

    new ImageJ();
    ClearCLIJ clij = new ClearCLIJ("HD");
    ElapsedTime.sStandardOutput = true;

    String file = "";
    if (System.getProperty("os.name").startsWith("Windows")) {
      file = "C:\\structure\\data\\cells.tif";
    } else {
      file = "/home/rhaase/data/Uncalibrated.tif";
    }

    ImagePlus imp = IJ.openImage(file);
    imp.show();

    //for (int i = 0; i < 1000; i++) {
      ElapsedTime.measure("the whole thing ", () -> {

        ClearCLImage input = clij.converter(imp).getClearCLImage();
        ClearCLImage output = clij.createCLImage(new long[]{input.getWidth()/2, input.getHeight()/2, input.getDepth()}, input.getChannelDataType());

        GPUSpotDetection gsd = new GPUSpotDetection(clij, input, output, threshold);
        gsd.setShowIntermediateResults(true);
        gsd.exec();

        System.out.println("spots: " + Kernels.sumPixels(clij, output));

        input.close();
        output.close();

      });
    //}
  }

}
