# The Dataset
This is a dataset that consists of a cleaned up subset of peer-to-peer loan data published by 
[LendingClub]((https://www.lendingclub.com/info/download-data.action)). We have built upon the results processed by the 
open-source [preprocess_lending_club_data](https://github.com/nateGeorge/preprocess_lending_club_data) repository, which
have been CC0-licensed on Kaggle [here](https://www.kaggle.com/wordsforthewise/lending-club). We processed the data 
into a smaller dataset format containing fewer rows and columns in order to more accurately reflect the real-world 
use-case of learning creditworthiness models from historical lending data.

The p2p_loans_470k dataset is intended to reflect a more real-world use case for machine learning than many typical
academic datasets. The training set contains over 300k rows detailing the outcome of real loans issued between
June 2012 through March 2015, while the test-set contains loans issued from April 2015 through August 2015. 

### Disclaimer
It should be noted that this dataset is not a perfect reflection of a real-world creditworthiness modeling use case. In 
a real use-case, models trained on fully-resolved historical lending data would be applied to loans issued years after 
the conclusion of the training set. In this data set, on the other hand, the test set is contiguous to the training set.
Nonetheless, the dataset spans years of real-world business practice, likely violating many traditional statistical 
assumptions of time-series stationarity.


# Instructions for use
The full dataset is already present as part of this repo. Simply copy the `p2p_loans_470k` directory to any place you
want the dataset.

If you want to rebuild the dataset yourself, you will need to have python3 and cmake installed. You will also need to 
install a Kaggle API token (see 
[https://github.com/Kaggle/kaggle-api#api-credentials](https://github.com/Kaggle/kaggle-api#api-credentials) for 
details). In the case that a new version of the base dataset is uploaded to Kaggle, it will be necessary to manually
download the old version of the dataset from Kaggle, as at the time of writing the Kaggle API does not appear to support
downloading previous versions of datasets.
 
Once you have completed this setup, simply run the default cmake command to install the proper dependencies, download 
the raw data, and run the processing script:

```bash
make
```

**Note:** If you want to use a virtualenvironment, create and activate your environment before running `make`.
