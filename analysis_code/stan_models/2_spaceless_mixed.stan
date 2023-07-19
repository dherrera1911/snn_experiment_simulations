data {
  int<lower=1> nNeurons;  // Number of groups
  int<lower=1> N; // Number of total datapoints
  vector[N] Stimulation; // Stimulation indicator
  vector[N] Response; // Response data
  array[N] int<lower=1> neuronId; // Neuron indices for each observation
}

parameters {
  real baselineFR;      // Average baseline FR
  real<lower=0> meanStimEffect;      // Increment of FR with stimulation
  vector[nNeurons] neuronBaseFR;  // Neuron-specific baseline FR
  vector[nNeurons] neuronEffect;  // Neuron-specific effect of stimulation
  real<lower=0> sigmaBaselineFR;     // Standard deviation of baseline FR
  real<lower=0> sigmaEffect;     // Standard deviation of stimulation effect
  real<lower=0> sigmaResidual;  // Standard deviation of residuals
  real<lower=0,upper=1> p;  // The probability that an arbitrary individual is in subpopulation 2
}

model {
  // Priors
  baselineFR ~ normal(0, 30);
  sigmaBaselineFR ~ cauchy(0, 5);
  neuronBaseFR ~ normal(baselineFR, sigmaBaselineFR);
  meanStimEffect ~ normal(0, 30);
  sigmaEffect ~ cauchy(0, 5);
  sigmaResidual ~ cauchy(0, 5);
  p ~ beta(1, 1);  // Prior for the mixture weight, assuming equal chance for both subpopulations.

  for (i in 1:N) {
    real lp_1 = neuronBaseFR[neuronId[i]];  // linear predictor without the effect
    real lp_2 = neuronBaseFR[neuronId[i]] + neuronEffect[neuronId[i]] * Stimulation[i];  // linear predictor with the effect

    // Mixture model for the response
    target += log_mix(p,
                      normal_lpdf(Response[i] | lp_1, sigmaResidual),
                      normal_lpdf(Response[i] | lp_2, sigmaResidual));
  }
}
