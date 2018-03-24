package de.mpicbg.spimcat.spotdetection.math;

public class Arrays {
    public static long[] multVectorScalar(long[] in, double scalar) {
        long[] out = new long[in.length];
        for (int i = 0; i < in.length; i++) {
            out[i] = (long)(in[i] * scalar);
        }
        return out;
    }

    public static long[] elementwiseMultiplVectors(long[] in1, float[]in2) {
        long[] out = new long[in1.length];
        for (int i = 0; i < in1.length; i++) {
            out[i] = (long)(in1[i] * in2[i]);
        }
        return out;
    }
}
