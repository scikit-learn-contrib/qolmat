rm(list=ls())
library(PKLMtest)

set.seed(42)

# Generate uniform random variables
unif_features <- replicate(100, runif(500, min = 0, max = 1))

empirical_cdf <- function(data) {
  sorted_data <- sort(data)
  y_values <- seq(1, length(data)) / length(data)
  return(list(x = sorted_data, y = y_values))
}

# Generate MCAR data

num_sim = 500
num_variables <- 5
num_samples <- 100
r <- 0.65
prob_nan <- 1 - r^(1/num_variables)

results_p_values <- numeric(num_sim)

for (i in 1:num_sim) {
  data <- matrix(runif(num_variables * num_samples, min = 0, max = 1), ncol = num_variables)
  mask <- matrix(runif(num_variables * num_samples) < prob_nan, ncol = num_variables)
  data[mask] <- NaN
  results_p_values[[i]] <- PKLMtest(
    data,
    num.proj = 100,
    nrep = 30,
    num.trees.per.proj = 200,
    size.resp.set = 2
  )
}

cdf_empirique_p_values <- ecdf(results_p_values)

# Plot
plot(
  cdf_empirique_p_values,
  main="Cumulative distribution function value of the p-values under H0",
  xlab="x: p_values under H0",
  ylab="F(x)",
  col="black",
  lwd=2,
  xlim = c(0, 1),
  ylim = c(0, 1)
)

for(i in 1:100) {
  cdf <- empirical_cdf(unif_features[, i])
  lines(cdf$x, cdf$y, col = rgb(0, 0, 1, alpha = 0.2))
}

abline(a = 0, b = 1, col = "red", lwd = 2)