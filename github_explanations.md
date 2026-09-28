EM-Algorithm: Examples
================
Patrick Mellady

- [Example 1: Mixture Distribution
  Problem](#example-1-mixture-distribution-problem)
  - [Distributional Problem
    Statement](#distributional-problem-statement)
    - [Finding the Necessary
      Quantities](#finding-the-necessary-quantities)
  - [Finding Derivatives to Maximize
    Iteratively](#finding-derivatives-to-maximize-iteratively)
  - [The Iterative Method](#the-iterative-method)
  - [Implementation and Results](#implementation-and-results)
    - [Results](#results)
- [Example 2: Missing Data](#example-2-missing-data)
  - [Distributional Statement](#distributional-statement)
  - [Finding Estimates with Complete
    Data](#finding-estimates-with-complete-data)
  - [Finding Estimates with Missing
    Data](#finding-estimates-with-missing-data)
  - [The Iterative Method](#the-iterative-method-1)
  - [Implementation](#implementation)
  - [Results](#results-1)

Here are a couple examples of using the EM algorithm on statistical
data. One example is a mixture of binomial distributions and the other
is an exercise 29 out of Casella and Berger chapter 7.

# Example 1: Mixture Distribution Problem

In this problem, we have a mixture of two binomial random variables with
mixture parameter $q$. These binomials share a size, $m$, but they have
success probabilities $p_1$ and $p_2$, respectively. The EM algorithm
maximizes the expected complete-data log likelihood, where the
expectation is taken over the latent variables given the data and the
previous iteration.

## Distributional Problem Statement

We start by specifying the distribution from which we obtain our sample.
With this distribution, we can maximize the incomplete log likelihood
via an iterative method. To do this, we assign latent variables
$Z_1, Z_2, \cdots, Z_n$ that satisfy $X_i|Z_i=1\sim bin(m, p_1)$. These
distributional statements give:

$$\begin{align*}
X_1, X_2, X_3, \cdots, X_n&\sim qbin(m,p_1)+(1-q)bin(m,p_2)\\
Z_1, Z_2, \cdots, Z_n&\overset{iid}{\sim}Bernoulli(q)\\
X_i|Z_i=1&\sim bin(m, p_1)\\
X_i|Z_i=0&\sim bin(m, p_2)
\end{align*}$$

where the pairs $(X_i,Z_i)$ are independent across $i$.

### Finding the Necessary Quantities

We will use the above distributional statements to find the posterior
distribution of $Z_i|X_i$ using Bayes’ rule.

$$\gamma_{1i}=P(Z_i=1|X_i)=\frac{P(X_i|Z_i=1)P(Z_i=1)}{P(X_i)}
            =\frac{{m\choose x_i}p_1^{x_i}(1-p_1)^{m-x_i}\cdot q}{{m\choose x_i}p_1^{x_i}(1-p_1)^{m-x_i}\cdot q+{m\choose x_i}p_2^{x_i}(1-p_2)^{m-x_i}\cdot (1-q)}$$

We will write this formula to obtain the $r^{th}$ iteration of
$\gamma_{1i}$, denoted $\hat{\gamma}_{1i}^{(r)}$, in the following way

$$\hat{\gamma}_{1i}^{(r)}=\frac{{m\choose x_i}(\hat{p}_1^{(r-1)})^{x_i}(1-\hat{p}_1^{(r-1)})^{m-x_i}\cdot \hat{q}^{(r-1)}}{{m\choose x_i}(\hat{p}_1^{(r-1)})^{x_i}(1-\hat{p}_1^{(r-1)})^{m-x_i}\cdot \hat{q}^{(r-1)}+{m\choose x_i}(\hat{p}_2^{(r-1)})^{x_i}(1-\hat{p}_2^{(r-1)})^{m-x_i}\cdot (1-\hat{q}^{(r-1)})}$$

Where $\hat p_j^{(r-1)}$ and $\hat q^{(r-1)}$ are the estimates for
$p_j$ and $q$ from iteration $r-1$. Now, to perform the EM, we will find
the expected log likelihood of our sample with respect to the latent
$z_i$’s. To simplify notation, we let

$$Q(\Theta|\Theta^{(r-1)})=E_{Z|X, \Theta^{(r-1)}}[\ln(L(X|p_1, p_2, q, Z))]$$

where $\Theta=(q, p_1, p_2)$. So we have

$$\begin{align*}
Q(\Theta|\Theta^{(r-1)})&=E_{Z|X, \Theta^{(r-1)}}[\ln(\prod_{i=1}^n[{m\choose x_i} p_1^{x_i} (1-p_1)^{m-x_i} q]^{z_i} \cdot[{m\choose x_i} p_2^{x_i} (1-p_2)^{m-x_i} (1-q)]^{1-z_i})]\\
                &=E_{Z|X, \Theta^{(r-1)}}[\sum_{i=1}^n[z_i[\ln{m\choose x_i}+x_i \ln(p_1)+(m-x_i)\ln(1-p_1)+\ln(q)]+\\
                &\qquad\qquad(1-z_i)[\ln{m\choose x_i}+x_i \ln(p_2)+(m-x_i)\ln(1-p_2)+\ln(1-q)]]]
\end{align*}$$

Now that we have the log-likelihood, we exploit the linearity of
expectation, together with $E[Z_i|X_i]=\hat{\gamma}_{1i}^{(r)}$ (since
$Z_i$ is an indicator), to obtain

$$\begin{align*}
Q(\Theta|\Theta^{(r-1)})&=\sum_{i=1}^n[\hat{\gamma}_{1i}^{(r)}[\ln{m\choose x_i}+x_i \ln(p_1)+\\
&(m-x_i)\ln(1-p_1)+\ln(q)]+(1-\hat{\gamma}_{1i}^{(r)})[\ln{m\choose x_i}+x_i \ln(p_2)+(m-x_i)\ln(1-p_2)+\ln(1-q)]]
\end{align*}$$

## Finding Derivatives to Maximize Iteratively

We can now find $\nabla_{\Theta} Q(\Theta|\Theta^{(r-1)})$ and maximize
over $\Theta$ to obtain $\Theta^{(r)}$. Note that, since
$\hat{\gamma}_{1i}^{(r)}$ is based on the $(r-1)^{th}$ iteration, it is
constant with respect to differentiation. We find the gradient by
component, starting with $q$:

$$\begin{align*}
\dfrac{\partial Q(\Theta|\Theta^{(r-1)})}{\partial q}&=\sum_{i=1}^n[\hat{\gamma}_{1i}^{(r)}\frac{1}{q}-(1-\hat{\gamma}_{1i}^{(r)})\frac{1}{1-q}]=0\\
&\implies\frac{1}{q(1-q)}\sum_{i=1}^n[\hat{\gamma}_{1i}^{(r)}(1-q)-(1-\hat{\gamma}_{1i}^{(r)})q]=0\\
&\implies\frac{1}{q(1-q)}\sum_{i=1}^n[\hat{\gamma}_{1i}^{(r)}-q]=0\\
&\implies\sum_{i=1}^n\hat{\gamma}_{1i}^{(r)}=nq\\
&\implies\frac{\sum_{i=1}^n\hat{\gamma}_{1i}^{(r)}}{n}=\hat{q}^{(r)}
\end{align*}$$

Next, we will take the derivative with respect to $p_1$, which gives

$$\begin{align*}
\dfrac{\partial Q(\Theta|\Theta^{(r-1)})}{\partial p_1}&=\sum_{i=1}^n[\hat{\gamma}_{1i}^{(r)}[\frac{x_i}{p_1} -\frac{m-x_i}{1-p_1}]]=0\\
&\implies\frac{1}{p_1(1-p_1)}\sum_{i=1}^n[\hat{\gamma}_{1i}^{(r)}[x_i(1-p_1) -(m-x_i)p_1]]=0\\
&\implies\sum_{i=1}^n[\hat{\gamma}_{1i}^{(r)}[x_i -mp_1]]=0\\
&\implies\sum_{i=1}^n\hat{\gamma}_{1i}^{(r)}x_i= \sum_{i=1}^n\hat{\gamma}_{1i}^{(r)}mp_1\\
&\implies\frac{\sum_{i=1}^n\hat{\gamma}_{1i}^{(r)}x_i}{m\sum_{i=1}^n\hat{\gamma}_{1i}^{(r)}}=\hat{p_1}^{(r)}
\end{align*}$$

From the symmetry of the problem, we find that

$$\frac{\sum_{i=1}^n(1-\hat{\gamma}_{1i}^{(r)})x_i}{m\sum_{i=1}^n(1-\hat{\gamma}_{1i}^{(r)})}=\hat{p_2}^{(r)}$$

## The Iterative Method

Now we have an iterative method:

- Initialize values for $\hat{q}, \hat{p}_1, \hat{p}_2$
- Use the values to find $\hat{\gamma}_{1i}$
- Use $\hat{\gamma}_{1i}$ to update $\hat{q}$, $\hat{p}_1$, $\hat{p}_2$
- Iterate to convergence

Lastly, all we must do is implement this code in R. The R code is in
this repository and covers data generation and running the algorithm.
The algorithm and results are shown below.

## Implementation and Results

Below, we implement the model described above

``` r
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
```

### Results

Plotting the data as well as a trace plot for each parameter gives the
following

``` r
# PLotting the data as well as the trace of our parameters
for_plot<-data.frame(x=X,
                     y=res$est[1]*dbinom(X, m, res$est[2])+
                       (1-res$est[1])*dbinom(X, m, res$est[3]))

ggplot(for_plot)+geom_histogram(aes(x, after_stat(density)))+
  geom_point(aes(x=x, y=y), color="red")+
  geom_segment(aes(x=x, y=0, xend=x, yend = y), color="red")+
  xlab("X")+ylab("Density")+labs(title="Observed Data and Model Implied Mixture")
```

    ## `stat_bin()` using `bins = 30`. Pick better value `binwidth`.

![](github_explanations_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

``` r
trace_df<-as.data.frame(cbind(1:res$iterations, res$params_trace))
colnames(trace_df)<-c("Iteration", "q", "p1", "p2")

(ggplot(trace_df)+geom_point(aes(x=Iteration, y=q))+ylab("Mixing Parameter")) /
  (ggplot(trace_df)+geom_point(aes(x=Iteration, y=p1, color="p1"))+
     geom_point(aes(x=Iteration, y=p2, color="p2"))+ylab("Probability")+
     scale_colour_manual(name="Parameter",
                         values=c(p1="orange", p2="blue")))
```

![](github_explanations_files/figure-gfm/unnamed-chunk-2-2.png)<!-- -->

# Example 2: Missing Data

The following problem is taken from Casella and Berger chapter 7
exercise 29. In this problem, we observe paired data $(X_i,Y_i)$ for
$i=1,2,\cdots,n$ where the $Y_i$ are independent with
$Y_i\sim pois(m\beta\tau_i)$, independent of
$(X_1,X_2,\cdots,X_n)\sim MN(m,\boldsymbol{\tau})$ with
$\boldsymbol{\tau}=(\tau_1,\tau_2,\cdots,\tau_n)$ and
$\sum_{i=1}^n\tau_i=1$ and $\sum_{i=1}^nx_i=m$. We first find the joint
mass function of $(Y,X)$.

## Distributional Statement

Since the multinomial mass function is
$f(x)=\frac{m!}{x_1!x_2!\cdots x_n!}\tau_1^{x_1}\tau_2^{x_2}\cdots\tau_n^{x_n}$
and the poisson mass function is
$f(y_i)=\frac{e^{-m\beta\tau_i}(m\beta\tau_i)^{y_i}}{y_i!}$, the
likelihood of $X$ and $Y$ is given by:

$$f(\textbf{y},\textbf{x}|\beta,\boldsymbol{\tau})=m!\prod_{i=1}^n\frac{e^{-m\beta\tau_i}(m\beta\tau_i)^{y_i}}{y_i!}\frac{\tau_i^{x_i}}{x_i!}$$

This gives us the following log-likelihood:

$$l=\ln(m!)+\sum_{i=1}^n[-m\beta\tau_i+y_i\ln(m\beta\tau_i)+x_i\ln(\tau_i)-\ln(y_i!x_i!)]$$

## Finding Estimates with Complete Data

First, we will find the maximum likelihood estimators by using the
complete data. This amounts to simply differentiating the log-likelihood
with respect to the parameters ($\beta,\tau_i$) and setting this equal
to zero to find estimates (for the $\tau_i$, we must respect the
constraint $\sum_{i=1}^n\tau_i=1$, which we do with a Lagrange
multiplier $\lambda$). We do this now:

$$\frac{\partial l}{\partial \beta}=\sum_{i=1}^n[-m\tau_i+\frac{y_i}{\beta}]=0\implies\hat{\beta}=\frac{\sum_{i=1}^ny_i}{m\sum_{i=1}^n\hat{\tau_i}}$$

However, recall that $\sum_{i=1}^n\hat{\tau}_i=1$ and that
$\sum_{i=1}^nx_i=m$, so the estimate above can be written as

$$\hat{\beta}=\frac{\sum_{i=1}^ny_i}{\sum_{i=1}^nx_i}$$

Now, we find the estimates for $\tau_i$

$$\frac{\partial}{\partial \tau_i}\Big[l-\lambda\Big(\sum_{j=1}^n\tau_j-1\Big)\Big]=-m\beta+\frac{y_i+x_i}{\tau_i}-\lambda=0$$

Multiplying by $\tau_i$ and summing over $i$, and using
$\sum_{i=1}^n\tau_i=1$, we get
$-m\beta+\sum_{i=1}^n[y_i+x_i]-\lambda=0$, so
$m\beta+\lambda=\sum_{i=1}^n[y_i+x_i]$. Solving the first-order
condition for $\tau_i$ gives
$\hat{\tau}_i=\frac{y_i+x_i}{m\hat\beta+\lambda}$, which gives us the
following estimate for $\tau_i$

$$\hat{\tau_i}=\frac{y_i+x_i}{\sum_{j=1}^n[y_j+x_j]}$$

## Finding Estimates with Missing Data

While the above is useful if we have all the data, we will suppose we do
not observe $\textbf{x}$ (a departure from the original problem). We
will use the above forms as a starting point for our analysis. See that
the estimates above depend on the unobserved data only through the
$x_i$:

$$\begin{align*}
\hat{\beta}&=\frac{\sum_{i=1}^ny_i}{\sum_{i=1}^nx_i}\\
\hat{\tau}_i&=\frac{y_i+x_i}{\sum_{j=1}^ny_j+\sum_{j=1}^nx_j}
\end{align*}$$

Since $\textbf{Y}$ is independent of $\textbf{X}$, we know that
$\textbf{X}|\textbf{Y}\sim MN(m,\boldsymbol{\tau})$, so, marginally,
$X_i\sim bin(m,\tau_i)$ and $E(X_i|\textbf{Y})=m\tau_i$. Heuristically,
we can guess that the missing data estimates replace $x_i$ with
$m\hat{\tau}_i^{(r)}$ (so that $\sum_{i=1}^nx_i$ is replaced by
$m\sum_{i=1}^n\hat{\tau}_i^{(r)}=m$), giving the form:

$$\begin{align*}
\hat{\beta}^{(r+1)}&=\frac{\sum_{i=1}^ny_i}{m}\\
\hat{\tau}^{(r+1)}_i&=\frac{y_i+m\hat{\tau_i}^{(r)}}{\sum_{j=1}^ny_j+m}
\end{align*}$$

We can make this process rigorous by reintroducing the conditional
expected log-likelihood that we saw in the first example. Let

$$Q(\Theta|\Theta^{(r)})=E_{X|Y,\Theta^{(r)}}(l)$$

where $\Theta=(\beta,\boldsymbol{\tau})$. The terms $\ln(m!)$ and
$\ln(y_i!x_i!)$ do not depend on $\Theta$, so they do not affect the
maximization. We will differentiate $Q$ with respect to our parameters.
See that

$$\begin{align*}
\frac{\partial Q}{\partial \beta}=\frac{\partial}{\partial \beta}E_{X|Y,\Theta^{(r)}}(l)=E_{X|Y,\Theta^{(r)}}(\frac{\partial l}{\partial \beta})&=E_{X|Y,\Theta^{(r)}}(\sum_{i=1}^n[-m\tau_i+\frac{y_i}{\beta}])\\
                                                                               &=E_{X|Y,\Theta^{(r)}}(-m\sum_{i=1}^n\tau_i+\frac{1}{\beta}\sum_{i=1}^ny_i)\\
                                                                               &=-m+\frac{1}{\beta}\sum_{i=1}^ny_i
\end{align*}$$

Setting this equal to zero and solving gives the following estimate

$$\hat{\beta}^{(r+1)}=\frac{\sum_{i=1}^ny_i}{m}$$

Similarly, for $\tau_j$, we again include the Lagrange multiplier
$\lambda$ for the constraint $\sum_{i=1}^n\tau_i=1$

$$\begin{align*}
\frac{\partial}{\partial \tau_j}\Big[Q-\lambda\Big(\sum_{i=1}^n\tau_i-1\Big)\Big]&=E_{X|Y,\Theta^{(r)}}(-m\beta+\frac{y_j+x_j}{\tau_j})-\lambda\\
                                                                                  &=-m\beta+\frac{y_j}{\tau_j}+\frac{E_{X|Y,\Theta^{(r)}}(x_j)}{\tau_j}-\lambda\\
                                                                                  &=-m\beta+\frac{y_j+m\hat{\tau}_j^{(r)}}{\tau_j}-\lambda
\end{align*}$$

Setting the above expression equal to zero, multiplying by $\tau_j$, and
summing over $j$ (using $\sum_{j=1}^n\tau_j=1$ and
$\sum_{j=1}^n\hat{\tau}_j^{(r)}=1$) gives
$m\beta+\lambda=\sum_{j=1}^ny_j+m$, so for every $j$ (including $j=1$)

$$\hat{\tau_j}^{(r+1)}=\frac{y_j+m\hat{\tau_j}^{(r)}}{\sum_{i=1}^ny_i+m}$$

## The Iterative Method

The above lends itself nicely to an iterative method:

- Initialize $\hat{\beta}^{(0)}$ and $\hat{\tau_j}^{(0)}$
- Calculate $\hat{\beta}^{(r+1)}$
- Calculate $\hat{\tau_j}^{(r+1)}$
- Iterate to convergence

## Implementation

We implement the missing data model below

``` r
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
  return(list(sol=sol, iter=k, error_tau=mean((truth$tau-tau)^2),
              error_beta=mean((truth$beta-beta)^2), ll=ll))
}


res<-EM_pois(X, Y, init, truth, .00001, 50000)
```

## Results

We observe the monotonically increasing expected log likelihood in the
following plot

``` r
ggplot()+geom_line(aes(x=1:res$iter, y=res$ll))+
  xlab("Iteration")+ylab("Log Likelihood")+labs(title="Log Likelihood Trace")
```

![](github_explanations_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

as well as examine how closely the parameters values match the truth
with via MSE below

``` r
mse_df<-round(data.frame(beta=res$error_beta, tau=res$error_tau),3)
kable(mse_df, caption="Mean Square Error for Estimated Parameters")
```

|  beta | tau |
|------:|----:|
| 0.014 |   0 |

Mean Square Error for Estimated Parameters
