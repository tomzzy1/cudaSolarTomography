#include <cassert>

#include "FITS_Image.hpp"


FITS_Image::FITS_Image(const struct FITS_Image::image& image)
    :
    bitpix{image.bitpix},
    any_null{image.any_null},
    null_value{image.null_value},
    naxes{image.naxes} {};


std::unique_ptr<FITS_Image>
load_FITS_image(const std::string& fts_fname) {
    fitsfile* fptr;
    int status = 0;
    assert(fits_open_image(&fptr, fts_fname.c_str(), READONLY, &status) == 0);

    float null_value = std::numeric_limits<float>::quiet_NaN();
    struct FITS_Image::image image;

    int naxis;
    int maxdim = 2;
    long _naxes[2];

    image.null_value = null_value;

    status = 0;
    assert(fits_get_img_param(fptr, maxdim, &image.bitpix, &naxis, _naxes, &status) == 0);
    assert(naxis == 2);

    image.naxes[0] = (size_t) _naxes[0];
    image.naxes[1] = (size_t) _naxes[1];

    std::unique_ptr<FITS_Image> fits_image;

    if (image.bitpix == FLOAT_IMG) {
        float* img = new float[image.naxes[0] * image.naxes[1]];

        int any_null = 0;
        assert(fits_read_img(fptr,
                             TFLOAT,
                             1,
                             image.naxes[0] * image.naxes[1],
                             &null_value,
                             img,
                             &any_null,
                             &status) == 0);
        fits_image = std::unique_ptr<FITS_Image>(new FloatFITS_Image(image, img));
    }
    else if (image.bitpix == DOUBLE_IMG) {
        double* img = new double[image.naxes[0] * image.naxes[1]];

        int any_null = 0;
        assert(fits_read_img(fptr,
                             TDOUBLE,
                             1,
                             image.naxes[0] * image.naxes[1],
                             &null_value,
                             img,
                             &any_null,
                             &status) == 0);
        fits_image = std::unique_ptr<FITS_Image>(new DoubleFITS_Image(image, img));
    }
    else {
        assert(false);
    }
    return fits_image;
}
