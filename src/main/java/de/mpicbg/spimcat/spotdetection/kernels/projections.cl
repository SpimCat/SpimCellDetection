__kernel void sum_project_3d_2d(
    write_only image2d_t dst,
    read_only image3d_t src
) {
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

  const int x = get_global_id(0);
  const int y = get_global_id(1);
  DTYPE_IN sum = 0;
  for(int z = 0; z < get_image_depth(src); z++)
  {
    sum = sum + READ_IMAGE(src,sampler,(int4)(x,y,z,0)).x;
  }
  WRITE_IMAGE(dst,(int2)(x,y),(DTYPE_OUT)sum);
}