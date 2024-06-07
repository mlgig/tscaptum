# tscaptum

tsCaptum is a library that adapt the popular post-hoc attribution methods implemented in the Captum
framework to work with time series by. Unlike previous library it is :
1) Fully compatible with scikit-learn and popular time-series frameworks as aeon and sktime
2) Take advantage of TS locality by using segmentation
3) it's extremely easy to use requiring almost no effort by the users

It's possible to install it by typing the command "pip install tsCaptum" or in case you have problems 
installing torch (e.g. you use Linux as OS and you want a lighter installation) use 
"pip3 install torch --index-url https://download.pytorch.org/whl/cpu && pip install tsCaptum"

In case you use the library please cite:

@misc{tsCaptum,
    
    author = {Davide Italo Serramazza, Thach Le Nguyen, Georgiana Ifrim},
    
    title = {tsCaptum: adapting Captum explainers for time series and scikit-learn-like predictors},
    
    howpublished = {GitHub},
    
    year = {2024},
    
    note = {Temporary bibitex entry},
    
    url = { https://github.com/mlgig/tscaptum },
}
