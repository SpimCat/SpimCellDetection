package de.mpicbg.spimcat.spotdetection.kernels;

import clearcl.ClearCLImage;
import clearcl.imagej.ClearCLIJ;
import de.mpicbg.spimcat.spotdetection.math.Arrays;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;

import java.util.HashMap;

public class GaussEff {

    public static void main(String... args) {
        new ImageJ();
        ClearCLIJ clij = new ClearCLIJ("HD");

        String file = "";
        if (System.getProperty("os.name").startsWith("Windows")) {
            file = "C:\\structure\\data\\Uncalibrated.tif";
        } else {
            file = "/home/rhaase/data/Uncalibrated.tif";
        }


        ImagePlus imp = IJ.openImage(file);
        imp.show();

        ClearCLImage src2 = clij.converter(imp).getClearCLImage();
        ClearCLImage dst = clij.converter(imp).getClearCLImage();

        long[] targetDimensions = Arrays.elementwiseMultiplVectors(src2.getDimensions(), new float[]{0.5f, 0.5f, 1});
        ClearCLImage src = clij.createCLImage(targetDimensions, src2.getChannelDataType());
        Kernels.crop(clij, src2, src, 256, 0, 0);


        gauss3d(clij, src, dst, 3, 3, 3);
    }

    public static void gauss3d(ClearCLIJ clij, ClearCLImage src, ClearCLImage dst, float sigmaX, float sigmaY, float sigmaZ) {
        gauss1dx(clij, src, dst, sigmaX);
    }
    private static void gauss1dx(ClearCLIJ clij, ClearCLImage src, ClearCLImage dst, float sigma) {
        float bs[] = computeBsFromSigma(sigma);
        float b = bs[0];
        float b1 = bs[1];
        float b2 = bs[2];
        float b3 = bs[3];

        float denom=1+b1+b2+b3;


        HashMap<String, Object> parameters = new HashMap<>();
        parameters.put("src", src);
        parameters.put("dst", dst);
        parameters.put("b", b);
        parameters.put("b1", b1);
        parameters.put("b2", b2);
        parameters.put("b3", b3);
        parameters.put("target_dim", 0);
        parameters.put("dim_x", 1);
        parameters.put("dim_y", 2);

        long[] globalSizes = new long[]{dst.getHeight(), dst.getDepth()};

        clij.execute(Kernels.class , "blur.cl", "gaussian_blur_young_2002_forward_image3d", globalSizes, parameters);

        clij.show(dst, "dst");

        // do it in x-direction

    }


    private static float[] computeBsFromSigma(float sigma) {
        //the code was borrowed from Geusebrock 2002 ( //http://staff.science.uva.nl/~mark/anigauss.c ) in great part
        float b;
        float b1;
        float b2;
        float b3;

        float q, qsq;
        float scale;

        /* initial values */
        float m0 = 1.16680f, m1 = 1.10783f, m2 = 1.40586f;
        float m1sq = m1*m1, m2sq = m2*m2;

        /* calculate q */
        if(sigma < 3.556)
            q = -0.2568f + 0.5784f * sigma + 0.0561f * sigma * sigma;
        else
            q = 2.5091f + 0.9804f * (sigma - 3.556f);

        qsq = q*q;

        /* calculate scale, and b[0,1,2,3] */
        scale = (m0 + q) * (m1sq + m2sq + 2*m1*q + qsq);
        b1 = -q * (2*m0*m1 + m1sq + m2sq + (2*m0 + 4*m1)*q + 3*qsq) / scale;
        b2 = qsq * (m0 + 2*m1 + 3*q) / scale;
        b3 = - qsq * q / scale;

        /* calculate B */
        b = (m0 * (m1sq + m2sq))/scale; b*=b;

        return new float[]{b, b1, b2, b3};
    }


// see header file for docs; 2002 version
    /*
    template <class VOXEL>
    void GaussIIR(Image3d <VOXEL> &img, const VOXEL SigmaX, const VOXEL SigmaY, const VOXEL SigmaZ) {

        //input/output image dimensions
    const signed long x_size = (signed long) img.GetWidth();
    const signed long y_size = (signed long) img.GetHeight();
    const signed long z_size = (signed long) img.GetNumSlices();
    const signed long s_size = x_size * y_size;     //just a helper variables (timesavers)
    const signed long x_2size = 2*x_size;
    const signed long x_3size = 3*x_size;
    const signed long s_2size = 2*s_size;
    const signed long s_3size = 3*s_size;

        if ((x_size > 1) && (x_size < 4)) {
            std::cerr << "GaussIIR(): Input image too small along x-axis, min is 4px.\n";
            return;
        }
        if ((y_size > 1) && (y_size < 4)) {
            std::cerr << "GaussIIR(): Input image too small along y-axis, min is 4px.\n";
            return;
        }
        if ((z_size > 1) && (z_size < 4)) {
            std::cerr << "GaussIIR(): Input image too small along z-axis, min is 4px.\n";
            return;
        }

        //setting the filter shape
        double b1D,b2D,b3D,BD;
        VOXEL B,b1,b2,b3;

        VOXEL *M=new VOXEL[9];		//boundary handling matrix
        if (M == NULL) {
            std::cerr << "GaussIIR(): No memory left for boundary matrix.\n";
            return;
        }

#ifdef I3D_DEBUG
        std::cout << "GaussIIR: input image of size " << x_size << "x" << y_size << "x" << z_size << " voxels\n";
#endif
        //temporary variables
        signed long i = 0, c = 0;

        VOXEL *wrk = img.GetFirstVoxelAddr();

        // ---- X AXIS ----
        if ((img.GetSizeX() > 1) && (SigmaX >= 1.0))
        {
            double S=static_cast<double>(SigmaX);
            computeBsFromSigma(S,BD,b1D,b2D,b3D);

            b1=static_cast<VOXEL>(b1D);
            b2=static_cast<VOXEL>(b2D);
            b3=static_cast<VOXEL>(b3D);
            B=static_cast<VOXEL>(BD);

            SuggestIIRBoundaries(-b1,-b2,-b3,M);
	const VOXEL denom=1+b1+b2+b3;	//boundary handling...

            //x-axis filter is present...
#ifdef I3D_DEBUG
            std::cout << "GaussIIR: convolving in x-axis...\n";
#endif
#ifdef I3D_DEBUG
            std::cout << "GaussIIR: B=" << B << ", b1=" << b1 << ", b2=" << b2 << ", b3=" << b3 << std::endl;
#endif
            for (signed long rows = 0; rows < y_size * z_size; rows++)
            {
                //no need to shift wrk pointer because x-lines are sequentially in memory

                //boundary handling...
	    const VOXEL bnd=*wrk/denom;

                // PREFIX, we do it "manually" (aka loop-unrolling)
#ifdef DEBUG_CORE_X
                std::cout << "GaussIIR:   forward: prefix starts, ";
#endif
                    *wrk=*wrk - bnd*(b1+b2+b3); //i=0
                REMOVE_REMAINDER_FORWARD(*wrk)
                ++wrk;

	    *wrk=*wrk  - *(wrk-1)*b1 - bnd*(b2+b3); //i=1
                REMOVE_REMAINDER_FORWARD(*wrk)
                ++wrk;

	    *wrk=*wrk  - *(wrk-1)*b1 - *(wrk-2)*b2 - bnd*b3; //i=2
                REMOVE_REMAINDER_FORWARD(*wrk)
                ++wrk;

#ifdef DEBUG_CORE_X
                std::cout << "convolutions of i=0,1,2 (row=" << rows << ") is finished\n";
#endif
                // INFIX
                for (i = 3; i < (x_size-1); i++)
                {
#ifdef DEBUG_CORE_X
                    std::cout << "GaussIIR:   forward: infix starts, ";
#endif
                        *wrk=*wrk  - *(wrk-1)*b1 - *(wrk-2)*b2 - *(wrk-3)*b3;
                    REMOVE_REMAINDER_FORWARD(*wrk)
#ifdef DEBUG_CORE_X
                    std::cout << "convolution of " << i << " (row=" << rows << ") is finished\n";
#endif
                        ++wrk;
                }

                ++wrk;
            }

            wrk=img.GetVoxelAddr(img.GetImageSize()-1); //useless

            for (signed long rows = 0; rows < y_size * z_size; rows++)
            {
                //no need to shift wrk pointer because x-lines are sequentially in memory

                VOXEL up=*wrk/denom; //a copy, btw: this equals to u_plus in (15) in [Triggs and Sdika, 2006]

	    *wrk=*wrk  - *(wrk-1)*b1 - *(wrk-2)*b2 - *(wrk-3)*b3; //finishing the forward convolution
                REMOVE_REMAINDER_FORWARD(*wrk)
                //the existence of (wrk-3) is guaranteed by the minimum image size test at the begining of this routine

                //boundary handling...
	    const VOXEL bnd1_=*wrk-up, //new *wrk values after the entire forward convolution are used now
                    bnd2_=*(wrk-1)-up,
                    bnd3_=*(wrk-2)-up;
                up/=denom; //this equals to v_plus in (15) in [Triggs and Sdika, 2006]

			*wrk=(M[0]*bnd1_ + M[1]*bnd2_ + M[2]*bnd3_ +up) *B;
	    const VOXEL bnd2=(M[3]*bnd1_ + M[4]*bnd2_ + M[5]*bnd3_ +up) *B;
	    const VOXEL bnd3=(M[6]*bnd1_ + M[7]*bnd2_ + M[8]*bnd3_ +up) *B;

                // PREFIX, we do it "manually" (aka loop-unrolling)
#ifdef DEBUG_CORE_X
                std::cout << "GaussIIR:   backward: prefix starts, ";
#endif
                //done already few lines above, i=0
                REMOVE_REMAINDER_BACKWARD(*wrk)
                --wrk;

	    *wrk=(*wrk)*B  - *(wrk+1)*b1 - bnd2*b2 - bnd3*b3; //i=1
                REMOVE_REMAINDER_BACKWARD(*wrk)
                --wrk;

	    *wrk=(*wrk)*B  - *(wrk+1)*b1 - *(wrk+2)*b2 - bnd2*b3; //i=2
                REMOVE_REMAINDER_BACKWARD(*wrk)
                --wrk;

#ifdef DEBUG_CORE_X
                std::cout << "convolutions of i=" << x_size-1 << "," << x_size-2 << "," << x_size-3 \
	    	      << " (row=" << rows << ") is finished\n";
#endif
                // INFIX
                for (i = 3; i < x_size; i++) //growing of i is unimportant in this case
                {
#ifdef DEBUG_CORE_X
                    std::cout << "GaussIIR:   backward: infix starts, ";
#endif
                        *wrk=(*wrk)*B  - *(wrk+1)*b1 - *(wrk+2)*b2 - *(wrk+3)*b3;
                    REMOVE_REMAINDER_BACKWARD(*wrk)
#ifdef DEBUG_CORE_X
                    std::cout << "convolution of " << x_size-1-i << " (row=" << rows << ") is finished\n";
#endif
                        --wrk;
                }
            }

            wrk=img.GetFirstVoxelAddr(); //important
        }

        // ---- Y AXIS ----
        if ((img.GetSizeY() > 1) && (SigmaY >= 1.0))
        {
            double S=static_cast<double>(SigmaY);
            computeBsFromSigma(S,BD,b1D,b2D,b3D);

            b1=static_cast<VOXEL>(b1D);
            b2=static_cast<VOXEL>(b2D);
            b3=static_cast<VOXEL>(b3D);
            B=static_cast<VOXEL>(BD);

            SuggestIIRBoundaries(-b1,-b2,-b3,M);
	const VOXEL denom=1+b1+b2+b3;	//boundary handling...

            //y-axis filter is present...
            VOXEL *wrk2 = wrk;

#ifdef I3D_DEBUG
            std::cout << "GaussIIR: convolving in y-axis...\n";
#endif
#ifdef I3D_DEBUG
            std::cout << "GaussIIR: B=" << B << ", b1=" << b1 << ", b2=" << b2 << ", b3=" << b3 << std::endl;
#endif
            //storage of original input values before any y-axis related convolution
            VOXEL *bnd=new VOXEL[x_size];

            for (signed long slices = 0; slices < z_size; slices++)
            {
                wrk = wrk2 + (slices * s_size);

                // PREFIX, we do it "manually" (aka loop-unrolling)
#ifdef DEBUG_CORE_Y
                std::cout << "GaussIIR:   forward: prefix starts, ";
#endif
                for (c = 0; c < x_size; c++) {
		*(bnd+c)=*(wrk+c)/denom; //boundary handling...

	    	*(wrk+c)=*(wrk+c) - *(bnd+c)*(b1+b2+b3); //i=0
                    REMOVE_REMAINDER_FORWARD(*(wrk+c))
                }
                wrk += x_size;

                for (c = 0; c < x_size; c++) {
	    	*(wrk+c)=*(wrk+c)  - *(wrk+c-x_size)*b1 - *(bnd+c)*(b2+b3); //i=1
                    REMOVE_REMAINDER_FORWARD(*(wrk+c))
                }
                wrk += x_size;

                for (c = 0; c < x_size; c++) {
	    	*(wrk+c)=*(wrk+c)  - *(wrk+c-x_size)*b1 - *(wrk+c-x_2size)*b2 - *(bnd+c)*b3; //i=2
                    REMOVE_REMAINDER_FORWARD(*(wrk+c))
                }
                wrk += x_size;

#ifdef DEBUG_CORE_Y
                std::cout << "convolutions of i=0,1,2 (slice=" << slices << ") is finished\n";
#endif
                // INFIX
                for (i = 3; i < (y_size-1); i++)
                {
#ifdef DEBUG_CORE_Y
                    std::cout << "GaussIIR:   forward: infix starts, ";
#endif
                    for (c = 0; c < x_size; c++) {
		    *(wrk+c)=*(wrk+c)  - *(wrk+c-x_size)*b1 - *(wrk+c-x_2size)*b2 - *(wrk+c-x_3size)*b3;
                        REMOVE_REMAINDER_FORWARD(*(wrk+c))
                    }
#ifdef DEBUG_CORE_Y
                    std::cout << "convolution of " << i << " (slice=" << slices << ") is finished\n";
#endif
                    wrk += x_size;
                }

                wrk += x_size;
            }

            wrk2=img.GetVoxelAddr(img.GetImageSize()) -x_size; //important

            VOXEL *bnd2=new VOXEL[x_size];
            VOXEL *bnd3=new VOXEL[x_size];
            VOXEL *vp=new VOXEL[x_size]; //to store v_plus in (15) in [Triggs and Sdika, 2006]

            VOXEL *Bnd2=new VOXEL[x_size];
            VOXEL *Bnd3=new VOXEL[x_size];

            for (signed long slices = 0; slices < z_size; slices++)
            {
                wrk = wrk2 - (slices * s_size);

                //copy the voxel's value at the end of y-line and finish the forward convolution
                for (c = 0; c < x_size; c++) {
		*(bnd+c)=*(wrk+c)/denom; //will be used *bnd instead of *up, just for this occassion

                    //finishes the forward convolution
		*(wrk+c)=*(wrk+c) - *(wrk+c-x_size)*b1 - *(wrk+c-x_2size)*b2 - *(wrk+c-x_3size)*b3;
                    REMOVE_REMAINDER_FORWARD(*(wrk+c))
                }

                for (c = 0; c < x_size; c++) *(vp+c)=*(bnd+c)/denom;
                for (c = 0; c < x_size; c++) *(bnd3+c)=*(wrk+c-x_2size)-*(bnd+c);
                for (c = 0; c < x_size; c++) *(bnd2+c)=*(wrk+c-x_size )-*(bnd+c);
                for (c = 0; c < x_size; c++) *(bnd +c)=*(wrk+c        )-*(bnd+c);

                //for (c = 0; c < x_size; c++) *(wrk +c)=( *(bnd+c)*M[0] + *(bnd2+c)*M[1] + *(bnd3+c)*M[2] + *(vp+c) ) *B;
                for (c = 0; c < x_size; c++) *(Bnd2+c)=( *(bnd+c)*M[3] + *(bnd2+c)*M[4] + *(bnd3+c)*M[5] + *(vp+c) ) *B;
                for (c = 0; c < x_size; c++) *(Bnd3+c)=( *(bnd+c)*M[6] + *(bnd2+c)*M[7] + *(bnd3+c)*M[8] + *(vp+c) ) *B;

                // PREFIX, we do it "manually" (aka loop-unrolling)
#ifdef DEBUG_CORE_Y
                std::cout << "GaussIIR:   backward: prefix starts, ";
#endif
                for (c = 0; c < x_size; c++) {
		*(wrk+c)=( *(bnd+c)*M[0] + *(bnd2+c)*M[1] + *(bnd3+c)*M[2] + *(vp+c) ) *B;
                    REMOVE_REMAINDER_BACKWARD(*(wrk+c))
                }
                wrk -= x_size;

                for (c = 0; c < x_size; c++) {
	        *(wrk+c)=*(wrk+c)*B  - *(wrk+c+x_size)*b1 - *(Bnd2+c)*b2 - *(Bnd3+c)*b3; //i=1
                    REMOVE_REMAINDER_BACKWARD(*(wrk+c))
                }
                wrk -= x_size;

                for (c = 0; c < x_size; c++) {
	        *(wrk+c)=*(wrk+c)*B  - *(wrk+c+x_size)*b1 - *(wrk+c+x_2size)*b2 - *(Bnd2+c)*b3; //i=2
                    REMOVE_REMAINDER_BACKWARD(*(wrk+c))
                }
                wrk -= x_size;

#ifdef DEBUG_CORE_Y
                std::cout << "convolutions of i=" << y_size-1 << "," << y_size-2 << "," << y_size-3 \
	    	      << " (slice=" << slices << ") is finished\n";
#endif
                // INFIX
                for (i = 3; i < y_size; i++)
                {
#ifdef DEBUG_CORE_Y
                    std::cout << "GaussIIR:   backward: infix starts, ";
#endif
                    for (c = 0; c < x_size; c++) {
		    *(wrk+c)=*(wrk+c) * B  - *(wrk+c+x_size)*b1 - *(wrk+c+x_2size)*b2 - *(wrk+c+x_3size)*b3;
                        REMOVE_REMAINDER_BACKWARD(*(wrk+c))
                    }
#ifdef DEBUG_CORE_Y
                    std::cout << "convolution of " << y_size-1-i << " (slice=" << slices << ") is finished\n";
#endif
                    wrk -= x_size;
                }
            }

            //release allocated memory
            delete[] bnd;
            delete[] bnd2;
            delete[] bnd3;
            delete[] vp;
            delete[] Bnd2;
            delete[] Bnd3;

            wrk=img.GetFirstVoxelAddr(); //important
        }

        // ---- Z AXIS ----
        if ((img.GetSizeZ() > 1) && (SigmaZ >= 1.0))
        {
            double S=static_cast<double>(SigmaZ);
            computeBsFromSigma(S,BD,b1D,b2D,b3D);

            b1=static_cast<VOXEL>(b1D);
            b2=static_cast<VOXEL>(b2D);
            b3=static_cast<VOXEL>(b3D);
            B=static_cast<VOXEL>(BD);

            SuggestIIRBoundaries(-b1,-b2,-b3,M);
	const VOXEL denom=1+b1+b2+b3;	//boundary handling...

            //z-axis filter is present...
            VOXEL *wrk2 = wrk;

#ifdef I3D_DEBUG
            std::cout << "GaussIIR: convolving in z-axis...\n";
#endif
#ifdef I3D_DEBUG
            std::cout << "GaussIIR: B=" << B << ", b1=" << b1 << ", b2=" << b2 << ", b3=" << b3 << std::endl;
#endif
            //storage of original input values before any y-axis related convolution
            VOXEL *bnd=new VOXEL[x_size];

            for (signed long cols = 0; cols < y_size; cols++)
            {
                wrk = wrk2 + (cols * x_size);

                // PREFIX, we do it "manually" (aka loop-unrolling)
#ifdef DEBUG_CORE_Z
                std::cout << "GaussIIR:   forward: prefix starts, ";
#endif
                for (c = 0; c < x_size; c++) {
		*(bnd+c)=*(wrk+c)/denom; //boundary handling...

	    	*(wrk+c)=*(wrk+c) - *(bnd+c)*(b1+b2+b3); //i=0
                    REMOVE_REMAINDER_FORWARD(*(wrk+c))
                }
                wrk += s_size;

                for (c = 0; c < x_size; c++) {
	    	*(wrk+c)=*(wrk+c)  - *(wrk+c-s_size)*b1 - *(bnd+c)*(b2+b3); //i=1
                    REMOVE_REMAINDER_FORWARD(*(wrk+c))
                }
                wrk += s_size;

                for (c = 0; c < x_size; c++) {
	    	*(wrk+c)=*(wrk+c)  - *(wrk+c-s_size)*b1 - *(wrk+c-s_2size)*b2 - *(bnd+c)*b3; //i=2
                    REMOVE_REMAINDER_FORWARD(*(wrk+c))
                }
                wrk += s_size;

#ifdef DEBUG_CORE_Z
                std::cout << "convolutions of i=0,1,2 (col=" << cols << ") is finished\n";
#endif
                // INFIX
                for (i = 3; i < (z_size-1); i++)
                {
#ifdef DEBUG_CORE_Z
                    std::cout << "GaussIIR:   forward: infix starts, ";
#endif
                    for (c = 0; c < x_size; c++) {
		    *(wrk+c)=*(wrk+c)  - *(wrk+c-s_size)*b1 - *(wrk+c-s_2size)*b2 - *(wrk+c-s_3size)*b3;
                        REMOVE_REMAINDER_FORWARD(*(wrk+c))
                    }
#ifdef DEBUG_CORE_Z
                    std::cout << "convolution of " << i << " (col=" << cols << ") is finished\n";
#endif
                    wrk += s_size;
                }

                wrk += s_size;
            }

            wrk2=img.GetVoxelAddr(img.GetImageSize()) -s_size;

            VOXEL *bnd2=new VOXEL[x_size];
            VOXEL *bnd3=new VOXEL[x_size];
            VOXEL *vp=new VOXEL[x_size]; //to store v_plus in (15) in [Triggs and Sdika, 2006]

            VOXEL *Bnd2=new VOXEL[x_size];
            VOXEL *Bnd3=new VOXEL[x_size];

            for (signed long cols = 0; cols < y_size; cols++)
            {
                wrk = wrk2 + (cols * x_size);

                //copy the voxel's value at the end of z-line and finish the forward convolution
                for (c = 0; c < x_size; c++) {
		*(bnd+c)=*(wrk+c)/denom; //will be used *bnd instead of *up, just for this occassion

                    //finishes the forward convolution
		*(wrk+c)=*(wrk+c) - *(wrk+c-s_size)*b1 - *(wrk+c-s_2size)*b2 - *(wrk+c-s_3size)*b3;
                    REMOVE_REMAINDER_FORWARD(*(wrk+c))
                }

                for (c = 0; c < x_size; c++) *(vp+c)=*(bnd+c)/denom;
                for (c = 0; c < x_size; c++) *(bnd3+c)=*(wrk+c-s_2size)-*(bnd+c);
                for (c = 0; c < x_size; c++) *(bnd2+c)=*(wrk+c-s_size )-*(bnd+c);
                for (c = 0; c < x_size; c++) *(bnd +c)=*(wrk+c        )-*(bnd+c);

                //for (c = 0; c < x_size; c++) *(wrk +c)=( *(bnd+c)*M[0] + *(bnd2+c)*M[1] + *(bnd3+c)*M[2] + *(vp+c) ) *B;
                for (c = 0; c < x_size; c++) *(Bnd2+c)=( *(bnd+c)*M[3] + *(bnd2+c)*M[4] + *(bnd3+c)*M[5] + *(vp+c) ) *B;
                for (c = 0; c < x_size; c++) *(Bnd3+c)=( *(bnd+c)*M[6] + *(bnd2+c)*M[7] + *(bnd3+c)*M[8] + *(vp+c) ) *B;

                // PREFIX, we do it "manually" (aka loop-unrolling)
#ifdef DEBUG_CORE_Z
                std::cout << "GaussIIR:   backward: prefix starts, ";
#endif
                for (c = 0; c < x_size; c++) {
		*(wrk+c)=( *(bnd+c)*M[0] + *(bnd2+c)*M[1] + *(bnd3+c)*M[2] + *(vp+c) ) *B;
                    REMOVE_REMAINDER_BACKWARD(*(wrk+c))
                }
                wrk -= s_size;

                for (c = 0; c < x_size; c++) {
	        *(wrk+c)=*(wrk+c)*B  - *(wrk+c+s_size)*b1 - *(Bnd2+c)*b2 - *(Bnd3+c)*b3; //i=1
                    REMOVE_REMAINDER_BACKWARD(*(wrk+c))
                }
                wrk -= s_size;

                for (c = 0; c < x_size; c++) {
	        *(wrk+c)=*(wrk+c)*B  - *(wrk+c+s_size)*b1 - *(wrk+c+s_2size)*b2 - *(Bnd2+c)*b3; //i=2
                    REMOVE_REMAINDER_BACKWARD(*(wrk+c))
                }
                wrk -= s_size;

#ifdef DEBUG_CORE_Z
                std::cout << "convolutions of i=" << z_size-1 << "," << z_size-2 << "," << z_size-3 \
	    	      << " (col=" << cols << ") is finished\n";
#endif
                // INFIX
                for (i = 3; i < z_size; i++)
                {
#ifdef DEBUG_CORE_Z
                    std::cout << "GaussIIR:   backward: infix starts, ";
#endif
                    for (c = 0; c < x_size; c++) {
		    *(wrk+c)=*(wrk+c) * B  - *(wrk+c+s_size)*b1 - *(wrk+c+s_2size)*b2 - *(wrk+c+s_3size)*b3;
                        REMOVE_REMAINDER_BACKWARD(*(wrk+c))
                    }
#ifdef DEBUG_CORE_Z
                    std::cout << "convolution of " << z_size-1-i << " (col=" << cols << ") is finished\n";
#endif
                    wrk -= s_size;
                }
            }

            //release allocated memory
            delete[] bnd;
            delete[] bnd2;
            delete[] bnd3;
            delete[] vp;
            delete[] Bnd2;
            delete[] Bnd3;
        }

        delete[] M;				//release boundary handling matrix
    }*/
}
