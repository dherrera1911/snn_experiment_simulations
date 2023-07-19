data {
  int<lower=1> nNeurons;  // Number of groups
  int<lower=1> N; // Number of total datapoints
  vector[N] Stimulation; // Stimulation indicator
  vector[N] Response; // Response data
  array[N] int<lower=1> neuronId; // Neuron indices for each observation
}

parameters {
  real baselineFR;      // Average baseline FR
  real meanStimEffect;      // Increment of FR with stimulation
  vector[nNeurons] neuronBaseFR;  // Neuron-specific baseline FR
  vector[nNeurons] neuronEffect;  // Neuron-specific effect of stimulation
  real<lower=0> sigmaBaselineFR;     // Standard deviation of baseline FR
  real<lower=0> sigmaEffect;     // Standard deviation of stimulation effect
  real<lower=0> sigmaResidual;  // Standard deviation of residuals
}

model {
  // Priors
  baselineFR ~ normal(0, 30);
  sigmaBaselineFR ~ cauchy(0, 5);
  neuronBaseFR ~ normal(baselineFR, sigmaBaselineFR);
  meanStimEffect ~ normal(0, 30);
  sigmaEffect ~ cauchy(0, 5);
  neuronEffect ~ normal(meanStimEffect, sigmaEffect);
  sigmaResidual ~ cauchy(0, 5);

  // Convert the above into loop form
  for (i in 1:N) {
    real mu = neuronBaseFR[neuronId[i]] + Stimulation[i] * neuronEffect[neuronId[i]];
    // real mu = condition == 0 ? baselineFR + neuronBaseFR[neuronId[i]] : stimEffect + neuronBaseFR[neuronId[i]];
    Response[i] ~ normal(mu, sigmaResidual);
  }
}
