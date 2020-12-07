data{
   int<lower=2, upper=4> K; //反応カテゴリ数
   int <lower=0> N; //回答者数
   int <lower=0> n; //項目数
   int<lower=1,upper=K> mu[N,n];
 }

 parameters {
   vector[N] theta;
   real<lower=0> alpha [n];
   ordered[K-1] kappa[n]; //カテゴリーの困難度
   real mu_kappa; //カテゴリーの困難度の事前分布の平均
   real<lower=0> sigma_kappa; //カテゴリーの困難度の事前分布の標準偏差
 }

 model{
   alpha ~ cauchy(0,5);
   theta ~ normal(0,1);
 for (i in 1: n){
   for (k in 1:(K-1)){
     kappa[i,k] ~ normal(mu_kappa,sigma_kappa);
   }}
 mu_kappa ~ normal(0,5);
 sigma_kappa ~ cauchy(0,5);
 for (i in 1:N){
  for (j in 1:n){
    mu[i,j] ~ ordered_logistic(theta[i]*alpha[j],kappa[j]);
 }}
 }
 
 generated quantities {
 vector[n] log_lik[N];
   for (i in 1: N){
     for (j in 1: n){
     log_lik[i, j] = ordered_logistic_lpmf (mu[i, j]|theta[i]*alpha[j],kappa[j]);
   }}
  }
