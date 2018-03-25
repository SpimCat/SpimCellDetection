
__kernel void gaussian_blur_young_2002_forward_image3d(
    write_only image3d_t dst, read_only image3d_t src,
      int target_dim, int dim_x, int dim_y,
      float b, float b1, float b2, float b3
)
{
    if (get_global_id(target_dim) < 3) {
        return;
    }
    int k_max = 0;
    if (target_dim == 0) {
        k_max = get_image_width(dst);
    } else if (target_dim == 1) {
       k_max = get_image_height(dst);
    } else  {
      k_max = get_image_depth(dst);
    }

    int4 pos = {0, 0, 0, 0};
    if (dim_x == 0) {
        pos.x = get_global_id(0);
    } else if (dim_x == 1) {
        pos.y = get_global_id(0);
    } else {
        pos.z = get_global_id(0);
    }
    if (dim_y == 0) {
        pos.x = get_global_id(1);
    } else if (dim_y == 1) {
        pos.y = get_global_id(1);
    } else {
        pos.z = get_global_id(1);
    }

    int4 pos_increment = {0, 0, 0, 0};
    if (target_dim == 0) {
        pos_increment.x = 1;
    } else if (target_dim == 1) {
        pos_increment.y = 1;
    } else {
        pos_increment.z = 1;
    }

    DTYPE_IN i0 = 0;
    DTYPE_IN i1 = 0;
    DTYPE_IN i2 = 0;
    DTYPE_IN i3 = 0;




    for (int k = 0; k < 4; k++) {
        i0 = i1;
        i1 = i2;
        i2 = i3;
        i3 = READ_IMAGE(src, sampler, pos).x;
        if (k > 3) {
            DTYPE_OUT value = i3 - i2*b1 - i1*b2 - i0*b3;
            WRITE_IMAGE(dst, pos, value);
        }
        pos = pos + pos_increment;
    }




}
