// adapted from Loic Royer, https://github.com/ClearVolume/ClearCL/blob/master/src/java/clearcl/ocllib/reduction/reductions.cl

__kernel
void reduce_sum_image_3d (__read_only image3d_t  src,
                          __write_only image1d_t  dst)
{
  const int width   = get_image_width(src);
  const int height  = get_image_height(src);
  const int depth   = get_image_depth(src);

  const int x       = get_global_id(0);
  const int y       = get_global_id(1);
  const int z       = get_global_id(2);

  const int stridex = get_global_size(0);
  const int stridey = get_global_size(1);
  const int stridez = get_global_size(2);

  float sum = 0;

  for(int lz=z; lz<depth; lz+=stridez)
  {
    for(int ly=y; ly<height; ly+=stridey)
    {
      for(int lx=x; lx<width; lx+=stridex)
      {
        const int4 pos = {lx,ly,lz,0};
        float value = (float)(READ_IMAGE(src, pos)).x;

        sum = sum + value;
      }
    }
  }

  int index = (x+stridex*y+stridex*stridey*z);

  WRITE_IMAGE(dst, (int4){index, 0,0,0}, sum);
}
