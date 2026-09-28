#################################
####### EM Algorithm ############
#################################
library(ggplot2)
library(patchwork)

# This file contains two examples of implementing the EM algorithm. The first 
# example is a mixture model of two binomial distributions where we augment
# the data with a latent variable that denotes from which distribution the data
# comes from. The second example is a missing data problem with Poisson regression.
# This is an exercise from Casella and Berger

################################################################################
# EM for Mixture Model Classification ##########################################
################################################################################

# Initializing the Algorithm
n<-1000
P.init<-c(.1, .6, .7)

# True Values
m<-20
q<-.4
p1<-.3
p2<-.9
P.true<-c(q, p1, p2)

# Generate Data
dat<-function(n, m, q, p1, p2){
  Z<-rbinom(n, 1, q)
  probs<-p1*Z+p2*(1-Z)
  X<-rbinom(n, m, probs)
  return(X)
}

# Define variable to calculate posterior probabilities
gamma1<-function(m, X, P){
  gam<-c()
  num<-P[1]*dbinom(X, m, P[2])
  denom<-P[1]*dbinom(X, m, P[2])+(1-P[1])*dbinom(X, m, P[3])
  return(num/denom)
}

# Likelihood function to track maximization
log_lik<-function(m, X, P){
  sum(log(P[1]*dbinom(X, m, P[2])+(1-P[1])*dbinom(X, m, P[3])))
}

# EM Algorithm Function
EM_binom<-function(X, P, epsilon=.00001, max_iters=5000){
  ll<-c(-Inf)
  params<-matrix(P, nrow=1)
  ll<-c(ll, log_lik(m, X, P))
  
  ll_1<-ll[1]
  ll_2<-ll[2]
  k<-2
  while(ll_1<ll_2){
    P.old<-params[k-1,]
    
    gamma1_n<-gamma1(m, X, P.old)
    gamma0_n<-1-gamma1(m, X, P.old)
    
    q_n<-sum(gamma1_n)/sum(gamma0_n+gamma1_n)
    p1_n<-sum(gamma1_n*X)/sum(m*gamma1_n)
    p2_n<-sum(gamma0_n*X)/sum(m*gamma0_n)
    
    params<-rbind(params, c(q_n, p1_n, p2_n))
    
    ll_1<-ll_2
    ll_2<-log_lik(m, X, params[k,])
    ll<-c(ll, ll_2)
    
    k<-k+1
    if(k%%100==0){
      print(k)
    }
    if(k>max_iters) break
    
  }
  
  est<-params[nrow(params),]
  return(list(est=est, iterations=k-1, ll=ll, params_trace=params))
}

X<-dat(n, m, q, p1, p2)
res<-EM_binom(X, P.init)

# PLotting the data as well as the trace of our parameters
for_plot<-data.frame(x=X,
                     y=res$est[1]*dbinom(X, m, res$est[2])+
                       (1-res$est[1])*dbinom(X, m, res$est[3]))

ggplot(for_plot)+geom_histogram(aes(x, after_stat(density)))+
  geom_point(aes(x=x, y=y), color="red")+
  geom_segment(aes(x=x, y=0, xend=x, yend = y), color="red")+
  xlab("X")+ylab("Density")+labs(title="Observed Data and Model Implied Mixture")

trace_df<-as.data.frame(cbind(1:res$iterations, res$params_trace))
colnames(trace_df)<-c("Iteration", "q", "p1", "p2")

(ggplot(trace_df)+geom_point(aes(x=Iteration, y=q))+ylab("Mixing Parameter")) /
  (ggplot(trace_df)+geom_point(aes(x=Iteration, y=p1, color="p1"))+
     geom_point(aes(x=Iteration, y=p2, color="p2"))+ylab("Probability")+
     scale_colour_manual(name="Parameter",
                         values=c(p1="orange", p2="blue")))


################################################################################
# EM for Missing Poisson Data ##################################################
################################################################################

# Initialize the true values
n<-20
m<-50
beta<-5
tau<-rep(1/n, n)
truth<-list(tau=tau, beta=beta)

# Generate the Data
Y<-rpois(n, m*beta*tau)
X<-rmultinom(1, m, tau)

# Initialize Starting Guesses for Parameters
t<-c(rep(1/(2*n), n/2 ), rep(3/(2*n), (n/2)))
b<-3
init<-list(tau=t, beta=b)

# EM algorithm Function
EM_pois<-function(X, Y, init, truth, epsilon=1e-8, lim=500000){
  # Create the likelihood stopping condition
  ## Log likelihood vector to track increase
  ll<-c()
  beta<-init$beta
  tau<-init$tau
  
  
  ## First imputation point
  beta<-sum(Y)/m
  tau<-(Y+m*tau)/(sum(Y)+m)
  ll<-c(ll, sum(dpois(Y, m*beta*tau, log=TRUE)))
  
  ## Second imputation point
  beta<-sum(Y)/m
  tau<-(Y+m*tau)/(sum(Y)+m)
  ll<-c(ll, sum(dpois(Y, m*beta*tau, log=TRUE)))
  
  ## Set iterations to 2, initiate the stopping condition
  k<-2
  l1<-ll[1]
  l2<-ll[2]
  while(l1<l2){
    beta<-sum(Y)/m
    tau<-(Y+m*tau)/(sum(Y)+m)

    l1<-l2
    l2<-sum(dpois(Y, m*beta*tau, log=TRUE))
    ll<-c(ll, l2)

    k<-k+1
    if(k%%50000==0){
      print(k)
    }
    
    if(k>lim) break
    
  }
  
  sol<-list(tau=tau, beta=beta)
  return(list(sol=sol, iter=k, error_tau=sum((truth$tau-tau)^2),
              error_beta=sum((truth$beta-beta)^2), ll=ll))
}


res<-EM_pois(X, Y, init, truth, .00001, 50000)

ggplot()+geom_line(aes(x=1:res$iter, y=res$ll))+
  xlab("Iteration")+ylab("Log Likelihood")+labs(title="Log Likelihood Trace")
