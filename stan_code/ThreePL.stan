data{
  int N; // 回答者数
  int n; // 項目数
  int<lower=0, upper=1> mu[N,n]; 
  // 各回答者の各項目への回答
}

parameters{
  real<lower=0, upper=5> a[n]; //識別度パラメータ
  real<lower=-4, upper=4> b[n]; // 困難度パラメータ
  real<lower=0, upper=1> c[n]; //  当て推量パラメータ
  real theta[N]; // 能力値
}

model{
  for(i in 1:N){
    for(j in 1:n){
      real p;
      p = inv_logit( a[j] * (theta[i] - b[j]));
      mu[i,j] ~ bernoulli( c[j] + (1-c[j])*p);
      }
    }

  /// 事前分布
  a ~ lognormal(0,1); 
  b ~ normal(0,1);  
  c ~ beta(5,23);  
  theta ~ normal(0,1);
}


generated quantities {
  vector[n] log_lik[N]; ///// 対数尤度
  int<lower=0, upper=1> mu_rep[N,n]; // 予測分布

  for(i in 1:N){
    for(j in 1:n){
      real p;
      p = inv_logit( a[j] * (theta[i] - b[j])) ;
      log_lik[i,j] = bernoulli_lpmf(mu[i,j] | c[j] + (1-c[j])*p );
      mu_rep[i,j] = bernoulli_rng( c[j] + (1-c[j]) *p );
    }
  }
}
