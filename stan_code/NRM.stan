data{
   int<lower=2, upper=4> K;
   int <lower=0> N;
   int <lower=0> n;
   int<lower=1,upper=K> mu[N,n];
 }
 
 parameters {
   vector[K] zeta[n]; //freely estimated intercept
   vector[K] lambda[n]; //freely estimated slope
   vector[N] theta;
 }
 
 transformed parameters {
   vector[K] zetan[n]; //intercept with constraints
   vector[K] lambdan[n]; //slope with constraints
   for (k in 1:n) {
     for (l in 1:K) {
       zetan[k,l]=zeta[k,l]-mean(zeta[k]); //constrain intercept sum for each item to 0
       lambdan[k,l]=lambda[k,l]-mean(lambda[k]); //constrain slope sum for each item to 0
     }}
 }
 
 model{
   theta ~ normal(0,1);
   for (i in 1: n){
   zeta[i] ~ normal(0,2);
   lambda[i] ~ normal(0,2);
 }
 for (i in 1:N)
   for (j in 1:n)
     mu[i,j] ~ categorical_logit(zetan[j] + lambdan[j]*theta[i]);
 }
 
 generated quantities {
   vector[n] log_lik[N];
   for (i in 1: N){
     for (j in 1: n){
       log_lik[i, j] = categorical_logit_lpmf (mu[i, j]| zetan[j] + lambdan[j]*theta[i]);
 }}
 
 }
 
 