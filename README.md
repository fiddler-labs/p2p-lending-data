# The Dataset
This is a dataset that consists of a cleaned up subset of peer-to-peer loan data published by 
[LendingClub]((https://www.lendingclub.com/info/download-data.action)). We have built upon the results processed by the 
open-source [preprocess_lending_club_data](https://github.com/nateGeorge/preprocess_lending_club_data) repository, which
 have been published on Kaggle [here](https://www.kaggle.com/wordsforthewise/lending-club).
 
We processed the data into a smaller dataset format containing fewer rows and columns in order to more accurately 
reflect the real-world use-case of learning creditworthiness models from historical lending data.


# Instructions for use
You will need to have python3 and cmake installed. You will also need to install a Kaggle API token (see 
[https://github.com/Kaggle/kaggle-api#api-credentials](https://github.com/Kaggle/kaggle-api#api-credentials) for 
details).
 
Once you have completed this setup, simply run the default cmake command to install the proper dependencies, download 
the raw data, and run the processing script:

```bash
make
```

NOTE: If you want to use a virtualenvironment, create and activate your environment before running `make`.