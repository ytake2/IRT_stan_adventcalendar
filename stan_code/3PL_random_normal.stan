data{
  int P;
  vector[P] a;
  vector[P] b;
  vector[P] c;
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
    real z;
    z=inv_logit( a[p] * (theta - b[p]));
    y[p] = bernoulli_rng( c[p] + (1-c[p]) *z);
  }
}
