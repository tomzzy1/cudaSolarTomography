#include <stdio.h>

#include "get_orbit.h"


int main(int argc, char** argv) {
    char idstring[] = "22272824pB.fts";
    double sun_ob[3];
    double carlong;
    double mjd;

    get_orbit(idstring, sun_ob, &carlong, &mjd);

    printf("sun_ob[0] = %f\n", sun_ob[0]);
    printf("sun_ob[1] = %f\n", sun_ob[1]);
    printf("sun_ob[2] = %f\n", sun_ob[2]);

    printf("carlong   = %f\n", carlong);

    printf("mjd       = %f\n", mjd);

    return 0;
}
