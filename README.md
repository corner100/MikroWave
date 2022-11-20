# Rohde & Schwarz Challange at TUM Hackathon
# Team µ~ (MicroWave)
![MikroWave](https://github.com/corner100/MikroWave/blob/main/images/Slide1.png)

# The Rohde & Schwarz QAR50
![MikroWave](https://github.com/corner100/MikroWave/blob/main/images/Slide2.png)

#
## We try to estimate the Reconstruction of the QAR50 and improve the measurement quality.
![MikroWave](https://github.com/corner100/MikroWave/blob/main/images/Slide3.png)

#
## We used simple shaped objects for the calibration of our algorithm.
![MikroWave](https://github.com/corner100/MikroWave/blob/main/images/Slide4.png)

#
## Automated Label Generation
![MikroWave](https://github.com/corner100/MikroWave/blob/main/images/Slide5.png)

#
## Estimation of the reconstruction filter by minimazing the MSError in the Fourierdomain of the magnitude
![MikroWave](https://github.com/corner100/MikroWave/blob/main/images/Slide6.png)

#
## Final Reconstruction Filter. High Frequenceys are amplified.
![MikroWave](https://github.com/corner100/MikroWave/blob/main/images/Slide7.png)
# Results
![MikroWave](https://github.com/corner100/MikroWave/blob/main/images/Slide8.png)

# How to use this repo

The training images are in:
https://github.com/corner100/MikroWave/tree/main/trainset

The Testset is in:
https://github.com/corner100/MikroWave/tree/main/testset/images

Run
https://github.com/corner100/MikroWave/tree/main/net.py
to reproduce our results.

You just need to generate a reconstruction filter ones and use it for every arbitrary object
