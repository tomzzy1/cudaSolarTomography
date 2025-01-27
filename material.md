# Tomography
- Jen Beatty, ``The Radon Transform and the Mathematics of Medical Imaging,'' 2012, [link](https://digitalcommons.colby.edu/cgi/viewcontent.cgi?article=1649&context=honorstheses). Concise and readable overview of tomography.
- Stanley R Deans, The Radon Transform and Some of Its Applications, 2007. Good textbook on the subject.
- [Link](https://web.eecs.umich.edu/~fessler/book/) to an unpublished textbook by U. of Michigan Prof. Jeff Fessler.

- Mark's pyinverse repo ([link](https://github.com/butala/pyinverse/tree/main)).
  * See the [notebook example](https://github.com/butala/pyinverse/blob/main/notebooks/Regularized%203D%20reconstruction.ipynb) demonstrating the iterative, regularized approached to 3-D tomographric reconstruction (which makes use of Radon trasnform matrices).

# Solar Tomography
- Altshculer and Perry, ``On determining the electron density distribution of the solar corona from K-coronameter data,'' Solar Physics, vol. 23, pp. 410-428, 1972. ([link](papers/BF00152315.pdf))
## Applications
- Phyics-based models of the solar wind / coronal mass ejections require a lower boundry condition on the electron density. I have collaborators in the [EUHFORIA (EUropean Heliospheric FORecasting Information Asset)](https://euhforia.com) who are potentially interested in using tomography-based estimates (they currently assume a constant electron density shell as their lower boundary condition).
## Coordinate Systems
- For good reason, the astronomy commuity is particularly careful about [coordinate systems](./papers/aa4262-05.pdf).

# Code Resources
- [Sunpy](https://sunpy.org/). This package is useful for downloading data, interpreting image data in the proper coordinate frame, manipulating data, e.g., preprocessing, and much more.
- [Astropy](https://www.astropy.org/). Sunpy is built on top of this package. Sometimes it is useful to refer Astropy documentation to learn how to correctly interact with quantities with units and coordinate systems.

# Data sources
The following list is fairly comprehensive but nowhere near complete. A cross-calibrated / curated dataset of more than one of the below data sources over even a month or year interval would be of great interest to the solar physics / imaging community.

## Ground Based
- COronal Solar Magnetism Observatory (COSMO) ([link](https://www2.hao.ucar.edu/mlso/instruments/cosmo-k-coronagraph-k-cor))
## Space Based
- Solar and Heliophysics Observatory (SOHO) LASCO C2 ([link](https://soho.nascom.nasa.gov/))
- Solar Dynamic Observatory (SDO) AIA ([link](https://sdo.gsfc.nasa.gov/))
- Solar Terrestrial Relations Observatory (STERO) SECHHI ([link](https://www.nasa.gov/mission_pages/stereo/main/index.html))
- Parker Solar Probe (PSP) WISPR ([link](https://www.nasa.gov/content/goddard/parker-solar-probe))
- Solar Orbiter METIS ([link](http://metis.oato.inaf.it/)) (though there are communicty concerns about data quality)


## Instructions to download data

The instructions below were provided by Alberto.

Getting C2 and C3 pB and BK calibrated images from the Legacy archive.

1. Navigate to:
   http://idoc-lasco.ias.u-psud.fr/sitools/client-portal/doc/
   A "Documentation & Data Access" list will appear.

2. Click on the 4th (from top) entry, "LASCO-C2 Scientific Products".
   The "DATA ACCESS" link will appear.
   Below it there is an explanation of the different accecible products:
   For pB images click the 3rd entry form the top.
   For BK images click the 4th entry from the top.

3. To access ANY product, first click the "DATA ACCESS" link.
   An interactive data-search form will load in the browser.
   It may take a while to fully load.

4. Click in the "Query Form Menu" at the top to open the science product menu.
   For pB images click the 2nd entry form the top ("Search Polariezed Radiance").
   For BK images click the 3rd entry from the top ("Search K Corona Radiance").
   In any case, the "Query Form" of the specified science product will show up.

5. In the "Query Form", select ta date range (either dates or Julian) and then click on SEARCH.
   This will open op ANOTHER window within the same browser, with the list of results.
   Move that window around as it surely covered up the "Query Form" already openned in step 4,
   which is there to be used for another search.
   A new window of results will open up for every search.
   You can maximize (or minimize, or close) any results open window at any time.
   For every available ofsevration you can donwload the FITS file, as well as a PNG visualization (very handy).

# Relevent Future Missions
- The Multiview Observatory for Solar Terrestrial Science (MOST) ([link](https://arxiv.org/abs/2303.02895#:~:text=MOST%20is%20envisioned%20as%20the,regions%2C%20and%20the%20solar%20wind.))
