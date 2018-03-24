package de.mpicbg.spimcat.spotdetection.tools;

import clearcl.ClearCLBuffer;
import clearcl.ClearCLHostImageBuffer;
import clearcl.ClearCLImage;
import clearcl.enums.HostAccessType;
import clearcl.enums.ImageChannelDataType;
import clearcl.enums.KernelAccessType;
import clearcl.imagej.ClearCLIJ;
import com.nativelibs4java.opencl.CLImageFormat;
import coremem.buffers.ContiguousBuffer;
import coremem.enums.NativeTypeEnum;
import de.mpicbg.spimcat.spotdetection.kernels.Kernels;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

import java.util.HashMap;

public class GPUSum {
    private ClearCLImage clImage;
    private ClearCLIJ clij;

    public GPUSum(ClearCLIJ clij, ClearCLImage clImage) {
        this.clij = clij;
        this.clImage = clImage;
    }

    public double sum() {

        ClearCLImage clReducedImage = clij.createCLImage(new long[]{clImage.getWidth(), clImage.getHeight()}, clImage.getChannelDataType());

        HashMap<String, Object> parameters = new HashMap<>();
        parameters.put("src", clImage);
        parameters.put("dst", clReducedImage);
        clij.execute(Kernels.class, "projections.cl", "sum_project_3d_2d", parameters);

        RandomAccessibleInterval rai = clij.converter(clReducedImage).getRandomAccessibleInterval();
        Cursor cursor = Views.iterable(rai).cursor();
        float sum = 0;
        while (cursor.hasNext()) {
            sum += ((RealType)cursor.next()).getRealFloat();
        }
        return sum;
    }
}
