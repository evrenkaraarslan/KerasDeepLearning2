# Removing gas flow meter from hardware with deep learning algorithm :hammer:

The main goal of creating this algorithm is detecting welding quality with deep learning. Wire arc additive manufacturing (WAAM) is required gas to be able to weld metals
and gas 
flow into to hose which connects welding machine and gas tank/bottle must
be known. However after training this algorithm with big data set, gas 
flow meter can be removed
from system hardware and be replaced with deep learning algorithm in real time.

**The training data has been collected with setting gas
pressure below minimum required bar to have bad results for training set and the rest of good
results have been taken from previous successful printing results with the pressure higher
than 3 bar. Csv files have limited information and data because of project privacy. Training algorithm and prediction algorithm can be seen in same script but It can be easily seperated. In stick out algorithm and in gas pressure algorithm, training and prediction is shown in the same script. This algorithm has been trained with 1 million line of training data in real project.**

The working principle of this code is mainly based on regression. The dataset inputs are currents and voltages which are same as stick out prediction
algorithm. However, the output disturibution is different than stick out algorithm. For
the sufficient gas pressure "0" has assigned as an output for the last column and when
the gas pressure does not sufficient for welding "1" has assigned for the output.

Some samples of training and prediction data can be seen below

Training

2019-11-22T10:35:21.3572426Z;20.01953;18.64624;0

2019-11-22T10:35:21.3572676Z;20.01953;18.61755;0

2019-12-04T12:49:14.0291276Z;95.64453;17.41272;1

2019-12-04T12:49:14.0291526Z;95.15625;17.38403;1

Prediction

2019-11-29T15:24:34.6218750Z;49.80469;16.75293

2019-11-29T15:24:34.6219000Z;50.78125;17.06848

2019-11-29T15:24:34.6219250Z;49.80469;17.01111

2019-11-29T15:24:34.6219500Z;50.29297;16.89636
