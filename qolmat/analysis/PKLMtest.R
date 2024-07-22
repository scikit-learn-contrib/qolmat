#' PKLMtest: compute a p-value for testing MCAR
#'
#' @param X a numeric matrix containing missing values encoded as NA, the data.
#' @param num.proj a positive integer specifying the number of projections to consider for the score.
#' @param num.trees.per.proj a positive integer, the number of trees per projection.
#' @param nrep a positive integer, the number of permutations.
#' @param min.node.size  a positive number, the minimum number of nodes in a tree.
#' @param size.resp.set an integer (>= 2), maximum number of classes allowed to be compared in each projection.
#' @param compute.partial.pvals a boolean, indicate if partial p-values shopuld be computed as well.
#' @param ... additional parameters.
#' @import ranger
#' @import parallel
#'
#' @examples
#' n <- 100
#' X <- cbind(rnorm(n),rnorm(n))
#' X.NA <- X
#' X.NA[,1] <- ifelse(stats::runif(n)<=0.2, NA, X[,1])
#'
#' pval <- PKLMtest(X.NA, num.proj = 5)
#'
#' @return a numeric value, the p-value(s) for the MCAR test, the first value is always the global p-value and if compute.partial.pvals is set to TRUE, the next values are the partial p-values for the relative importance of each variable.
#'
#' @export
PKLMtest <- function(X,
                     num.proj = 300,
                     num.trees.per.proj = 10,
                     nrep = 500,
                     min.node.size = 10,
                     size.resp.set = 2,
                     compute.partial.pvals = FALSE,
                     ...) {

  # checks
  if (num.proj < 2) {
    stop("num-proj should be at least 2.")
  }

  if (size.resp.set < 2) {
    stop("size.resp.set should be at least 2.")
  }

  # simplification
  #size.resp.set <- size.resp.set - 1

  # checks
  if (!is.matrix(X)) {
    stop("X should be a matrix.")
  }

  # get the mask for the data provided in X
  M <- is.na(X)

  # compute the permutations of the missing value patterns
  Mperm <- lapply(1:nrep, function(j) M[sample(nrow(M)),])

  # list of results
  list.obs <- list()
  list.null <- list()
  list.var.cov <- list()
  list.var.resp <- list()

  # main loop
  # Équivalent d'une boucle for sur les projections
  while (length(list.obs) != num.proj) {

    # sample a projection for the response
    new.var <- c()
    num.classes <- 1
    while (length(new.var)==0 ||  num.classes <= 1 ) {

      # sample a projection for the covariates

      # Définit la taille de l'espace A
      size.cov <- sample(1:ncol(X), size = 1, replace = FALSE)
      # Tire l'ensemble A
      var.cov <- sample(1:ncol(X), size = size.cov, replace = FALSE)

      # Regarde les variables qui pourraient être dans B
      new.var <- setdiff(1:ncol(X), var.cov)

      if (length(new.var)>1) {

        if (size.resp.set=="random"){ # On ne considère pas ce cas là pour le moment
          size.resp <- sample(1:length(new.var), size = 1, replace = FALSE)
        }else{ # Dans le cas où size.resp.set = 2, on tire tjrs un couple
          size.resp<- sample(1:min(length(new.var), size.resp.set), size=1, replace=FALSE)
        }
        # Tire l'ensemble B
        var.resp <- sample(new.var, size = size.resp, replace = FALSE)
      } else if (length(new.var)==1) {
        var.resp <- new.var
      } else if (length(new.var)==0){
        var.resp <- var.cov
      }

      # sélectionne les lignes de X projettées sur A totalement observées ie l'espace NA du papier
      # which one are complete there
      ids.keep <- which(apply(X[,var.cov,drop=FALSE],
                              1, function(x) !any(is.na(x))))

      # Projette M sur NA, B
      # define the new response
      M.resp <- M[ids.keep,var.resp,drop=F]
      # Selection des patterns uniques
      unique.pat.resp <- unique(M.resp)
      # Compte le nombre de patterns
      num.classes<- nrow(unique(M.resp))

    }

    # Si A et B sont "biens" tirés, alors on les ajoute à la liste des projs et on continue
    list.var.cov[[length(list.var.cov)+1]] <- var.cov
    list.var.resp[[length(list.var.resp)+1]] <- var.resp

    # Identifie les différents pattern pour créer les classes
    ids.patterns.resp <- sapply(
        1:nrow(unique.pat.resp),
        function(i) which(
            apply(
                M.resp,
                1,
                function(m) identical(
                    m,
                    unique.pat.resp[i,]
                )
            )
        ),
        simplify = F
    )

    # get the data for the projection on NA B
    X.proj <- X[ids.keep,var.cov,drop=F]

    # create the class for the projection
    class.resp <- rep(NA, length(ids.keep))

    # Pour chaque ligne de X.proj, on créée le label pour la tâche de classif
    for (i in 1:length(ids.patterns.resp)) {
      class.resp[ids.patterns.resp[[i]]] <- i
    }

    # Pour chaque permutation recrée les labels
    class.resp.perm<-lapply(1:nrep, FUN=function(j){


      M.respj <- Mperm[[j]][ids.keep,var.resp,drop=F]
      unique.pat.respj <- unique(M.respj)
      num.classesj<- nrow(unique(M.respj))


      ids.patterns.resp <- sapply(1:nrow(unique.pat.respj),
                                  function(i) which(apply(M.respj,
                                                          1,
                                                          function(m) identical(m,
                                                                                unique.pat.respj[i,]))), simplify = F)

      # create the class for the projection
      class.respj <- rep(NA, length(ids.keep))

      for (i in 1:length(ids.patterns.resp)) {
        class.respj[ids.patterns.resp[[i]]] <- i
      }

      return(class.respj)


    } )
    # Créée le dataset pour la tâche de classif
    d <- data.frame(y = factor(class.resp),
                    X = X.proj)

    # Entraînement du modèle
    st <- tryCatch({ranger::ranger(y~., data = d,
                                   num.trees = num.trees.per.proj,
                                   classification = TRUE,
                                   min.node.size = min.node.size,
                                   probability = TRUE)
    }, error = function(e) NA)
    # ?
    if(!any(is.na(st))) {

      # generate the observed stat
      # class.resp contient un vecteur avec toutes les étiquettes du dataframe
      obs <- genU(st, lab = class.resp)
      # generate the stat under H0
      null <- sapply(class.resp.perm, function(p) genU(st, lab = p))
      list.obs[[length(list.obs)+1]] <- obs
      list.null[[length(list.null)+1]] <- null

    }else{
      list.var.cov <- list.var.cov[-length(list.var.cov)]
      list.var.resp <- list.var.cov[-length(list.var.resp)]
    }





  }
  print("while done")
  # Concaténation en ligne  
  stat.perm.raw <- Reduce(list.null,f = rbind)




  isIn <- sapply(1:ncol(X), function(i) sapply(list.var.resp, function(l) !(i %in% l))) # list.var.resp
  isIn <- cbind(rep(TRUE, num.proj ), isIn)
  if (!compute.partial.pvals) {
    isIn <- isIn[,1,drop=FALSE]
  }

  partial.perm <- apply(isIn ,2, function(x) {
    ids <- which(x)

    colMeans(stat.perm.raw[ids,,drop=FALSE], na.rm=T)

  })

  if (ncol(isIn)==1) {
    partial.perm <- matrix(partial.perm, nrow=1)
  }

  partial.obs <- apply(isIn ,2, function(x) {

    if (sum(x)==0) {
      return(NA)
    } else {
      ids <- which(x)
      return(mean(unlist(list.obs[ids]),
                  na.rm=T))
    }


  })

  ##### partial p-values #########

  if (compute.partial.pvals){
    pvals <- sapply(1:(ncol(X)+1), function(i) (sum(partial.perm[,i] >= partial.obs[i])+1)/(length(partial.perm[,i])+1))
    names(pvals) <- c("all", paste0("-",1:ncol(X)))
  }else{
    pvals<-(sum(partial.perm >= partial.obs)+1)/(length(partial.perm)+1)
    names(pvals) <- "all"
  }

  ##############################

  return(pvals)

}