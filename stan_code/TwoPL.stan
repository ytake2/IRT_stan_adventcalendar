data{
  int N; //number of subjects
  int n; // number of items
  int<lower=0, upper=1> x[N,n]; // binary data matrix
}

parameters{
  real<lower=0, upper=5> a[n];
  real<lower=-4, upper=4> b[n]; // item difficulties
  real theta[N]; // latent factor scores
}

model{
  for(i in 1:N){// MIRT formulation
    for(j in 1:n){
      x[i,j] ~ bernoulli_logit(a[j] * (theta[i] - b[j]) );
  }
}
  b ~ normal(0,1);  /// difficulty prior
  a ~ lognormal(0,1); /// or lognormal(0,2)
  theta ~ normal(0,1); // latent factor
}

// calculate the correlation from the cholesky
generated quantities {
  vector[n] log_lik[N]; ///// row log likelihood
  real dev; /////// deviance
  real log_lik0; ///// global log likelihood
  vector[N] log_lik_row;
  int<lower=0, upper=1> x_rep[N,n]; // binary data matrix
  for(i in 1:N){// MIRT formulation
    for(j in 1:n){
      log_lik[i,j] = bernoulli_logit_lpmf(x[i,j] | a[j] * (theta[i] - b[j]) );
      x_rep[i,j] = bernoulli_logit_rng(a[j] * (theta[i] - b[j]) );
    }
  }
  for(i in 1:N){
    log_lik_row[i] = sum(log_lik[i]);
  }
  log_lik0 = sum(log_lik_row); // global log-likelihood
  dev = -2*log_lik0; // model deviance
}
