# Source code for training the neural live-polyline

This code is the basis of results in the paper (*Supporting Semi-Automatic Marble Thin-Section Image Segmentation with Machine Learning*) submitted to EAIS 2018.

### Structure of the code

Folder msrt contains the code for measuring the metrics for live-polyline and for running the MCBW algorithm. The file msrt.py is responsible for starting these measurements. There are 5 modes:

1. Beta measurements (calculate the betas for each curve, did not appeared in the article)
2. Error rate (for a given weight map this can calculate the error rate)
3. Validation (error rate on a single curve)
4. Theoretical error rate (calculate the error rates from the betas on an image)
5. Simulation (the MCBW algorithm to calculate the error rates from the ps, pt values, theoretical error in the article).

Otherwise the training is managed by livepoly.py. Example for starting the pre-training:

```bash
python livepoly.py --mode 0 --memory 0.3 --iteration 30000 --lr 0.0001 --epochs 10
```

### Installation

In order to run the algorithm you should create a [python environment](http://python-guide-pt-br.readthedocs.io/en/latest/dev/virtualenvs/) and then install the required packages by typing **pip install -r requirements.txt**. You can find the txt file among the source files. This will use Tensorflow with GPU.
