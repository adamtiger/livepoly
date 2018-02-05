# Source code for training the neural live-polyline

This code is the basis of results in the paper submitted to EAIS 2018.

### Structure of the code

Folder msrt contains the code for measuring the metrics for live-polyline and for running the MCBW algorithm. The file msrt.py is responsible for starting these measurements. There are 5 modes:

1. Beta measurements (calculate the betas for each curve, did not appeared in the article)
2. Error rate (for a given weight map this can calculate the error rate)
3. Validation (error rate on a single curve)
4. Theoretical error rate (calculate the error rates from the betas on an image)
5. Simulation (the MCBW algorithm to calculate the error rates from the ps, pt values, theoretical error in the article).

Otherwise the training is managed by livepoly.py. 
