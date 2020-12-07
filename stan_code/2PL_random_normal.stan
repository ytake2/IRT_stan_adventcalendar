data{
  int P;
  vector[P] a;
  vector[P] b;
}

parameters{
  real temp;
}

model{
  temp ~ uniform(0,1);
}

generated quantities{
  int y[P];
  real theta;
  theta = normal_rng(0,1);
  for(p in 1:P){
    y[p] = bernoulli_logit_rng(a[p]*(theta-b[p]));
  }
}
