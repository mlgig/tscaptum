# tsCaptum

tsCaptum is a library that adapts the popular post-hoc attribution methods implemented in the Captum
framework to work with time series. Unlike previous libraries tsCaptum is :
1) Fully compatible with scikit-learn and popular time-series frameworks such as aeon and sktime
2) Takes advantage of TS locality by using time series segmentation 
3) It's extremely easy to use requiring almost no effort by the users

It can be installed by typing the command "pip install tsCaptum" or in case you have problems 
installing torch (e.g. you use Linux as OS and you want a lighter installation) use 
"pip3 install torch --index-url https://download.pytorch.org/whl/cpu && pip install tsCaptum"

In case you use this library please cite:

@misc{tsCaptum,
    
    author = {Davide Italo Serramazza, Thach Le Nguyen, Georgiana Ifrim},
    title = {tsCaptum: adapting Captum explainers for time series and scikit-learn-like predictors},
    howpublished = {GitHub},
    year = {2024},
    note = {Temporary bibitex entry},
    url = { https://github.com/mlgig/tscaptum },
}
