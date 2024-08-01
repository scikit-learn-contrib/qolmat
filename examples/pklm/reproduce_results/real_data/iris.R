install.packages("mice")
library("mice")
library(PKLMtest)
# Chargez le dataset iris
data(iris)

# Affichez les premières lignes du dataset pour vérifier son contenu
head(iris)

# Convertissez le dataset en une matrice
iris_matrix <- as.matrix(iris[, 1:4])  # Ne prenez que les quatre premières colonnes numériques

# Affichez les premières lignes de la matrice pour vérifier
head(iris_matrix)

iris.na <- ampute(data = iris_matrix, prop = 0.3, mech = "MAR")$amp

pv <- PKLMtest(
  matrix(unlist(iris.na), ncol = 4, byrow = TRUE),
  num.proj = 300,
  nrep = 30,
  num.trees.per.proj = 200,
  size.resp.set = 2
)
print(pv)