/* get_orbit.c
 *
 * Get the sun --> observer vector (km) and the carrington
 *    longitude (in DEG) of the observer.
 *
 *  In the case of EUVI .fts files have the orbit parameters in them,
 *    we can just read them and avoid the rest of the code.  See
 *    more comments below in the EUVI section.
 *
 * If the orbit file is found in ORBIT_FILE_DIR it won't look on the web.
 * If it is not found there, it will look on the web in ORBIT_URL and
 *   will save the retrieved orbit file to ORBIT_SRATCH_DIR
 *
 * by Paul Janzen and Richard Frazin Summer/Fall 1999 modified by
 *   Butala in 2004 and Frazin in 2006,7,8,2010
 *
 * Modified by A.M.Vásquez, CLASP Fall-2017, to include WISPR,
 *                          deal with new LAM LASCO-C2 headers,
 *                          deal with pre-processed KCOR headers (1),
 *                          and also comments for documentation.
 *
 * NOTE (1): Our own kcor_prep.pro IDL tool should be used.
 *
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

//#include <fitsfile.h>
//#include "headers.h"

#include "libwcs/fitsfile.h"

#include "get_orbit.h"
#include "r3misc.h"


#define C2BUILD
#define MARSEILLES

#define DATADIR "../../../../data/lasco_c2/"
#define ORBIT_FILE_DIR "../../../../data/orbits/"
#define ORBIT_URL_VERSION_MAX 9

#define MAXPATH 256

#define NDATES 164


#define ALPHApo (286.13 * M_PI/180.0)	/* J2000 solar pole coords */
#define DELTApo (63.87 * M_PI/180.0)


void get_orbit(char *idstring, double *sun_ob, double *carlong, double *mjd) {
 char fd[14], frcr[5];
 char sdate[] = "xxxx-xx-xx.xxxx", hr[] = "xx", mn[] = "xx";
 int i, k, kmin, fdt, version, local_orbit_file;
 double ersun[3], erscr[3], o[3], e[3], p[3], c[3], xx[3];
 double jd, sjd, jdmin, diff, abdiff, frcdy, eclong;
 double result = 0;
 char buffer[MAXPATH], bigline[512], orbfn[MAXPATH];
 FILE *fid_orb;
#ifdef ORBIT_URL
 char orburl[MAXPATH], wget_command[MAXPATH], scratch[MAXPATH];
#endif

	/*  The COR, EUVI, AIA, and WISPR .fts files give the Sun-Spacecraft vector in
	 *  Heliocentric Aries Ecliptic Coordinates.  This differs
	 *  from GCI in the origin point and choice of Z-axis (ecliptic
	 *  N, vs. equatorial N (celestial pole).  Therefore these coords.
	 *  need to rotated about the x-axis.
	 */
#if (defined CORBUILD || defined EUVIBUILD || defined AIABUILD || defined WISPRIBUILD || defined WISPROBUILD || defined KCOR)
 {
  char *header, *fitsdate;
  int lhead, nbhead;
  Rot *Rx;

  strcpy(buffer,DATADIR);
  strcat(buffer,idstring);
  assert((header = fitsrhead(buffer, &lhead, &nbhead)) != NULL);

  // Get Sun_Ob [m] in the HAE CS, as pointer "c".
  assert(hgetr8(header,"HAEX_OBS",c));
  assert(hgetr8(header,"HAEY_OBS",c+1));
  assert(hgetr8(header,"HAEZ_OBS",c+2));
  // Convert Sun_Ob HAE to [km].
  for (k = 0;k < 3;k++)
    c[k] *= 0.001;

  // Get Sub-Spacecraft Carrington Longitude [deg], as pointer "carlong".
  assert(hgetr8(header,"CRLN_OBS",carlong));

  fitsdate = hgetc(header,"DATE_OBS");

  *mjd = fd2mjd(fitsdate);
  fprintf(stdout,"get_orbit.c: datestring from FITS file is: ");
  fprintf(stdout,"%s, modified julian date is: %.8g\n",fitsdate,*mjd);

  /* the J2000.0 angle between the Ecliptic and mean Equatorial planes
   * is 23d26m21.4119s - From Allen's Astrophysical Quantities, 4th ed. (2000) */

#ifdef KCOR
  // If dealing with KCOR data, do NOT rotate, simply set: sun_ob = HAE_OBS = DSUN [1,0,0],
  // as it is only used in build_subA (or compare.c) to get DSUN from its norm.
  r3eq(sun_ob,c);
#else
  // Compute now Sun_Ob [km] in CS-1 (J2000), as vector "sun_ob".
  Rx = rotx(0.40909262920459);
  rotvmul(sun_ob,Rx,c);
  free(Rx);
#endif

 return;
 }
#endif

 /* solar meridian crossing times from the ephemeris */
 char fdl[NDATES][14] =
   { "1995-12-20.72", "1996-01-17.06", "1996-02-13.40", "1996-03-11.73",
     "1996-04-08.03", "1996-05-05.28", "1996-06-01.50", "1996-06-28.70",
     "1996-07-25.90", "1996-08-22.13", "1996-09-18.39", "1996-10-15.67",
     "1996-11-11.97", "1996-12-09.28", "1997-01-05.61", "1997-02-01.95",
     "1997-03-01.29", "1997-03-28.60", "1997-04-24.87", "1997-05-22.10",
     "1997-06-18.31", "1997-07-15.50", "1997-08-11.72", "1997-09-07.97",
     "1997-10-05.24", "1997-11-01.54", "1997-11-28.84", "1997-12-26.17",
     "1998-01-22.50",
     "1998-02-18.85", "1998-03-18.17", "1998-04-14.46", "1998-05-11.70",
     "1998-06-07.91",
     "1998-07-05.11", "1998-08-01.32", "1998-08-28.56", "1998-09-24.82",
     "1998-10-22.10",
     "1998-11-18.41", "1998-12-15.72", "1999-01-12.06", "1999-02-08.40",
     "1999-03-07.73",
     "1999-04-04.04", "1999-05-01.30", "1999-05-28.52", "1999-06-27.72",
     "1999-07-21.92",
     "1999-08-18.15", "1999-09-14.40", "1999-10-11.68", "1999-11-07.97",
     "1999-12-05.28",
     "2000-01-01.61", "2000-01-28.95", "2000-02-25.29", "2000-03-23.61",
     "2000-04-19.89",
     "2000-05-17.12", "2000-06-13.33", "2000-07-10.53", "2000-08-06.74",
     "2000-09-02.98",
     "2000-09-30.25", "2000-10-27.54", "2000-11-23.85", "2000-12-21.17",
     "2001-01-17.50",
     "2001-02-13.84", "2001-03-13.17", "2001-04-09.47", "2001-05-06.72",
     "2001-06-02.94",
     "2001-06-30.13", "2001-07-27.34", "2001-08-23.57", "2001-09-19.83",
     "2001-10-17.11",
     "2001-11-13.41", "2001-12-10.73", "2002-01-07.06", "2002-02-03.40",
     "2002-03-02.73",
     "2002-03-30.04", "2002-04-26.31", "2002-05-23.54", "2002-06-19.74",
     "2002-07-16.94",
     "2002-08-13.96", "2002-09-09.41", "2002-10-06.68", "2002-11-02.98",
     "2002-11-30.29",
     "2002-12-27.61", "2003-01-23.95", "2003-02-20.29", "2003-03-19.61",
     "2003-04-15.90",
     "2003-05-13.14", "2003-06-09.35", "2003-07-06.55", "2003-08-02.76",
     "2003-08-29.99",
     "2003-09-26.26", "2003-10-23.55", "2003-11-19.85", "2003-12-17.17",
     "2004-01-13.50",
     "2005-01-02.51", "2005-01.29.39", "2005-02-25.72", "2005-04-21.32",
     "2005-05-18.56",
     "2004-02-09.84", "2006-05-08.16", "2006-06-04.37", "2006-07-01.57",
     "2006-07-28.77", "2006-08-25.00", "2006-09-21.26", "2006-10-18.55",
     "2006-11-14.85", "2006-12-12.16", "2007-01-08.49", "2007-02-04.83",
     "2007-03-04.17", "2007-03-31.48", "2007-04-27.75", "2007-05-24.98",
     "2007-06-21.18", "2007-07-18.38", "2007-08-14.60", "2007-09-10.84",
     "2007-10-08.12", "2007-11-04.42", "2007-12-01.72", "2007-12-29.05",
     "2008-01-25.39", "2008-02-21.73", "2008-03-20.05", "2008-04-16.34",
     "2008-05-13.58", "2008-06-09.78", "2008-07-06.98", "2008-08-03.19",
     "2008-08-30.43", "2008-09-26.70", "2008-10-23.98", "2008-11-20.29",
     "2008-12-17.61", "2009-01-13.94", "2009-02-10.28", "2009-03-09.61",
     "2009-04-05.92", "2009-05-03.17", "2009-05-30.39", "2009-06-26.59",
     "2009-07-23.79", "2009-08-20.02", "2009-09-16.27", "2009-10-13.55",
     "2009-11-09.85", "2009-12-07.17"
   };

 /* carrington longitude calculation for Earth */
#if (defined C2BUILD || defined C3BUILD)
 char *header, *fitsdate, *fitstime;
  int lhead, nbhead;
  strcpy(buffer,DATADIR);
  strcat(buffer,idstring);
  assert((header = fitsrhead(buffer, &lhead, &nbhead)) != NULL);
#ifdef NRL
 strncpy(hr, idstring + 15, 2);
 strncpy(mn, idstring + 17, 2);
#endif
#ifdef MARSEILLES
 fitstime = hgetc(header,"TIME_OBS");
 fprintf(stderr,"fitstime = %s\n",fitstime);
 strncpy(hr, fitstime    , 2);
 strncpy(mn, fitstime + 3, 2);
#endif
frcdy = atof(hr) / 24.0 + atof(mn) / (24.0 * 60.0);
#elif defined EITBUILD
 strncpy(hr, idstring + 12, 2);
 strncpy(mn, idstring + 14, 2);
 frcdy = atof(hr) / 24.0 + atof(mn) / (24.0 * 60.0);
#endif

 fdt = (int) rint(10000 * frcdy);
 sprintf(frcr, "%04d", fdt);

#if (defined C2BUILD || defined C3BUILD)
#ifdef NRL
 strncpy(sdate, idstring + 6, 4);
 strncpy(sdate + 5, idstring + 10, 2);
 strncpy(sdate + 8, idstring + 12, 2);
#endif
#ifdef MARSEILLES
 fitsdate = hgetc(header,"DATE_OBS");
 fprintf(stderr,"fitsdate = %s\n",fitsdate);
 strncpy(sdate    , fitsdate     , 4);
 strncpy(sdate + 5, fitsdate + 5 , 2);
 strncpy(sdate + 8, fitsdate + 8 , 2);
#endif
#elif defined EITBUILD
 strncpy(sdate, idstring + 3, 4);
 strncpy(sdate + 5, idstring + 7, 2);
 strncpy(sdate + 8, idstring + 9, 2);
#endif

 strncpy(sdate + 11, frcr, 4);
 *mjd = fd2mjd(sdate);
 fprintf(stderr,"get_orbit.c: sdate is: ");
 fprintf(stderr,"%s, modified julian date is: %.8g\n",sdate,*mjd);

 fprintf(stdout,"get_orbit.c: sdate is: ");
 fprintf(stdout,"%s, modified julian date is: %.8g\n",sdate,*mjd);

 sjd = fd2jd(sdate);
 abdiff = 60.0;
 diff = abdiff;

 for (k = 0; k < NDATES; k++) {
   strncpy(fd, fdl[k], 13);
   strcat(fd, "\0");
   jd = fd2jd(fd);
   if (fabs(jd - sjd) < abdiff) {
     abdiff = fabs(jd - sjd);
     diff = jd - sjd;
     jdmin = jd;
     kmin = k;
   }
 }

 if (abdiff == 60.0) {
   fprintf(stderr, "get_orbit: date is not in range!!!  ");
   fprintf(stderr, "sdate =%s\n", sdate);
   exit(666);
 } else {
   eclong = 360.0 * diff / 27.2753;
 }

 /* k is the line number in the orbit file */
 k = (int) rint(frcdy * 144.0);
 if (k == 144)
   k = 143;
 if (k > 144) {
   fprintf(stderr, "get_orbit: frcdy > 1.0!!!!\n");
   exit(69);
 }

 version = -1; /* orbit file version number */
 fid_orb = NULL;

 local_orbit_file = 0;
#ifdef ORBIT_FILE_DIR
 /*open orbit file.  Loop over file version numbers */
 while ((fid_orb == NULL)
     && (version <= ORBIT_URL_VERSION_MAX)){
    strcpy(orbfn, ORBIT_FILE_DIR);
    strcat(orbfn, "SO_OR_PRE_");
#if (defined C2BUILD || defined C3BUILD)
#ifdef NRL
    strncat(orbfn, idstring + 6, 8);
#endif
#ifdef MARSEILLES
 strncpy(sdate    , fitsdate     , 4);
 strncpy(sdate + 4, fitsdate + 5 , 2);
 strncpy(sdate + 6, fitsdate + 8 , 2);
 strncat(orbfn, sdate, 8);
#endif

#elif defined EITBUILD
    strncat(orbfn, idstring + 3, 8);
#endif
    version++;
    strcat(orbfn, "_V0");
    sprintf(buffer, "%d", version);
    strcat(orbfn, buffer);
    strcat(orbfn, ".DAT");
    fprintf(stderr, "orbfn =%s\n", orbfn);
    fid_orb = fopen(orbfn, "r");
 }
#endif

 if (fid_orb != NULL)
   local_orbit_file = 1;

 /* If the orbit file isn't found (or there is ORBIT_FILE_DIR isn't defined)
  *   AND if there is no URL to search then
  *   exit with an error.  Otherwise try the URL.  */
#ifndef ORBIT_URL
 if (fid_orb == NULL) {
   fprintf(stderr,
    "get_orbit: orbit file %s not found!\n", orbfn);
   exit(0);
 }
#endif

 /* if the files are not in the yearly directory and are one below
  *  comment out the first two lines after the ifdef  */
#ifdef ORBIT_URL
 version = -1;
 while (fid_orb == NULL && version <=  ORBIT_URL_VERSION_MAX) {
   strcpy(orburl, ORBIT_URL);
   strcpy(scratch, ORBIT_SCRATCH_DIR);
   strcpy(orbfn, "SO_OR_PRE_");
#if (defined C2BUILD || defined C3BUILD)
#ifdef NRL
   strncat(orburl,idstring + 6, 4);
   strcat(orburl,"/");
   strncat(orbfn, idstring + 6, 8);
#endif
#ifdef MARSEILLES
 fitsdate = hgetc(header,"DATE_OBS");
 fprintf(stderr,"fitsdate = %s\n",fitsdate);
 strncpy(sdate , fitsdate , 4);
 strncat(orburl, sdate    , 4);
 strcat(orburl,"/");
 strncpy(sdate + 4, fitsdate + 5 , 2);
 strncpy(sdate + 6, fitsdate + 8 , 2);
 strncat(orbfn, sdate, 8);
 fprintf(stderr,"orburl = %s\n",orburl);
 fprintf(stderr,"orbfn  = %s\n",orbfn);
#endif

#elif defined EITBUILD
   strncat(orburl,idstring + 3, 4);
   strcat(orburl,"/");
   strncat(orbfn, idstring + 3, 8);
#endif

   version++;
   strcat(orbfn, "_V0");
   sprintf(buffer, "%d", version);
   strcat(orbfn, buffer);
   strcat(orbfn, ".DAT");
   strcat(scratch,orbfn);

   strcpy(wget_command, WGETCOMMAND);
   strcat(wget_command, ORBIT_SCRATCH_DIR);
   strcat(wget_command," ");
   strcat(wget_command, orburl);
   strcat(wget_command, orbfn);

   if(system(wget_command) != 0) {
     /* Fetch was not successful.
      * If wget creates a size 0 output file even for unsuccessful fetches
      *    delete the size 0 file.  Os X does not have this issue, so the
      *    remove statement has been disabled. */
     /* assert(remove(orbfn) == 0); */
   }
   fid_orb = fopen(scratch, "r"); /*open newly downloaded file */
 }

 if (fid_orb == NULL){
   fprintf(stderr,"\nget_orbit.c: looked for file: %s\n   could not execute:  \n%s\n",orbfn, wget_command);
   assert(0);
 } else {
   if (local_orbit_file == 1) {
       fprintf(stderr,"local orbit file retrieved: %s\n",orbfn);
       fflush(stderr);
     } else {
       fprintf(stderr,"orbit file retrieved from %s%s\n",orburl,orbfn);
       fflush(stderr);
   }
 }

#endif

 for (i = 0; i <= k; i++)
   fgets(bigline, 512, fid_orb);
 fclose(fid_orb);

 /* GCI earth-sun vector */
 sscanf(bigline + 305, "%lf %lf %lf", &ersun[0], &ersun[1], &ersun[2]);
 /* GCI earth - SOHO vector */
 sscanf(bigline + 44, "%lf %lf %lf", &erscr[0], &erscr[1], &erscr[2]);

#if (defined C2BUILD || defined EITBUILD || defined C3BUILD)
 /* C2, C3 and EIT instruments are on SOHO */
 for (i = 0; i < 3; i++)
   sun_ob[i] = -(ersun[i] - erscr[i]);

#elif defined MK4BUILD
 /* MK4 instrument is on Earth */
 for (i = 0; i < 3; i++)
   sun_ob[i] = -ersun[i];
#endif

 /*calculate angle between earth and soho, projected onto equ. plane */

 // Solar Pole vector in CS-1:
 p[0] = cos(DELTApo) * cos(ALPHApo);
 p[1] = cos(DELTApo) * sin(ALPHApo);
 p[2] = sin(DELTApo);

 /* e = -ersun - p*(p'*(-ersun))
  * o = sun_ob - p*(p'*(sun_ob)) */
 r3eq(e, ersun);
 r3scalmul(e, -1.0);
 r3eq(o, sun_ob);

 r3eq(xx, p);
 r3scalmul(xx, -1.0 * r3dot(p, o));
 r3add(o, o, xx);
 r3norm(o);

 r3eq(xx, p);
 r3scalmul(xx, -1.0 * r3dot(p, e));
 r3add(e, e, xx);
 r3norm(e);

 /* take the cross product to get angular difference
  * recall that x cross y = z */
 r3cross(c, e, o);
 result = asin(r3dot(p, c)) * 180.0 / M_PI;

 /*add the angular difference to get carlong */
 *carlong = eclong + result;
 return;
}

void get_orbit_wrapper(int year, int month, int day,
                      int hour, int minute,
		       double *sun_ob, double *carlong, double *mjd) {
 char buffer[MAXPATH];

#if (defined C3BUILD || defined C2BUILD || defined EITBUILD)
 sprintf(buffer, "XX-XX-%d%02d%02d_%02d%02d", year, month, day, hour, minute);
#endif

 get_orbit(buffer, sun_ob, carlong, mjd);
}
