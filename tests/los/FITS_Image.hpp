#pragma once

#include <string>
#include <array>

#include "fitsio.h"


/* C programmers should note that the ordering of arrays in FITS
 * files, and hence in all the CFITSIO calls, is more similar to the
 * dimensionality of arrays in Fortran rather than C. For instance if
 * a FITS image has NAXIS1 = 100 and NAXIS2 = 50, then a 2-D array
 * just large enough to hold the image should be declared as
 * array[50][100] and not as array[100][50].
 */

/* From Section 3.3.2 of the FITS standard version 3.0

   Arrays of more than one dimension shall consist of a sequence such
   that the index along axis 1 varies most rapidly, that along axis 2
   next most rapidly, and those along subsequent axes progressively
   less rapidly, with that along axis m, where m is the value of
   NAXIS, varying least rapidly.
 */


/* The data are stored column major.
 *
 * For COR1, note that
 * CTYPE1  = 'HPLN-TAN'           /
 * CTYPE2  = 'HPLT-TAN'           /
 *
 * For column major, axis 1, longitude, changes more rapidly than axis
 * 2. Axis 1 corresponds to the horizontal axis. So, the data are
 * stored column major and the horizontal axis changes more rapidly
 * than the vertical axis. So, the data do NOT need to be read in
 * FORTRAN order.
 */


class FITS_Image {
public:
    struct image {
        bool any_null;
        int bitpix;
        float null_value;
        std::array<size_t, 2> naxes;
    };

    const int bitpix;

    const bool any_null;
    const float null_value;
    const std::array<size_t, 2> naxes;

    FITS_Image(const struct image& image);

    virtual ~FITS_Image() {}

    virtual double operator[](const size_t i) = 0;
};


class FloatFITS_Image : public FITS_Image {
public:
    FloatFITS_Image(const struct image& image, float* img)
        : FITS_Image(image) { this->img = img; }

    ~FloatFITS_Image() { delete[] img; }

    double operator[](const size_t i) { return (double) img[i]; }

private:
    float* img;
};


class DoubleFITS_Image : public FITS_Image {
public:
    DoubleFITS_Image(const struct image& image, double* img)
        : FITS_Image(image) { this->img = img; }

    ~DoubleFITS_Image() { delete[] img; }

    double operator[](const size_t i) { return img[i]; }

private:
    double* img;
};


std::unique_ptr<FITS_Image>
load_FITS_image(const std::string& fts_fname);
