---
title: 💀 Домашка
nav_order: 3
---

## Matrix calculus

1. Find the gradient $$\nabla f(x)$$ and hessian $$f''(x)$$, if $$f(x) = \dfrac{1}{2} \|Ax - b\|^2_2$$.
1. Find gradient and hessian of $$f : \mathbb{R}^n \to \mathbb{R}$$, if:

    $$
    f(x) = \log \sum\limits_{i=1}^m \exp (a_i^\top x + b_i), \;\;\;\; a_1, \ldots, a_m \in \mathbb{R}^n; \;\;\;  b_1, \ldots, b_m  \in \mathbb{R}
    $$
1. Calculate the derivatives of the loss function with respect to parameters $$\frac{\partial L}{\partial W}, \frac{\partial L}{\partial b}$$ for the single object $$x_i$$ (or, $$n = 1$$)
![](../images/simple_learning.svg)
1. Calculate: $$\dfrac{\partial }{\partial X} \sum \text{eig}(X), \;\;\dfrac{\partial }{\partial X} \prod \text{eig}(X), \;\;\dfrac{\partial }{\partial X}\text{tr}(X), \;\; \dfrac{\partial }{\partial X} \text{det}(X)$$
1. Calculate the first and the second derivative of the following function $$f : S \to \mathbb{R}$$
	$$
	f(t) = \text{det}(A − tI_n),
	$$
	where $$A \in \mathbb{R}^{n \times n}, S := \{t \in \mathbb{R} : \text{det}(A − tI_n) \neq 0\}	$$.
1. Find the gradient $$\nabla f(x)$$, if $$f(x) = \text{tr}\left( AX^2BX^{-\top} \right)$$.

## Automatic differentiation
You can use any automatic differentiation framework in this section (Jax, PyTorch, Autograd etc.)

1. You will work with the following function for this exercise,

	$$
	f(x,y)=e^{−\left(sin(x)−cos(y)\right)^2}
	$$
	
	Draw the computational graph for the function. Note, that it should contain only primitive operations - you need to do it automatically -  [jax example](https://bnikolic.co.uk/blog/python/jax/2022/02/22/jax-outputgraph-rev.html), [PyTorch example](https://github.com/waleedka/hiddenlayer) - you can google/find your own way to visualise it.

1. Compare analytic and autograd (with any framework) approach for the hessian of:		
	
	$$
	f(x) = \dfrac{1}{2}x^TAx + b^Tx + c
	$$

1. Suppose, we have the following function $$f(x) = \frac{1}{2}\|x\|^2$$, select a random point $$x_0 \in \mathbb{B}^{1000} = \{0 \leq x_i \leq 1 \mid \forall i\}$$. Consider $$10$$ steps of the gradient descent starting from the point $$x_0$$:

	$$
	x_{k+1} = x_k - \alpha_k \nabla f(x_k)
	$$

	Your goal in this problem is to write the function, that takes $$10$$ scalar values $$\alpha_i$$ and return the result of the gradient descent on function $$L = f(x_{10})$$. And optimize this function using gradient descent on $$\alpha \in \mathbb{R}^{10}$$. Suppose, $$\alpha_0 = \mathbb{1}^{10}$$.

	$$
	\alpha_{k+1} = \alpha_k - \beta \frac{\partial L}{\partial \alpha}
	$$

	Choose any $$\beta$$ and the number of steps your need. Describe obtained results.

1. Compare analytic and autograd (with any framework) approach for the gradient of:		
	
	$$
	f(X) = - \log \det X
	$$

1. Compare analytic and autograd (with any framework) approach for the gradient and hessian of:		
	
	$$
	f(x) = x^\top x x^\top x
	$$

## Convex sets

1. Show that the convex hull of the $$S$$ set is the intersection of all convex sets containing $$S$$.
1. Show that the set of directions of the strict local descending of the differentiable function in a point is a convex cone.
1. Prove, that if $$S$$ is convex, then $$S+S = 2S$$. Give an counterexample in case, when $$S$$ - is not convex.
1. Let $$x \in \mathbb{R}$$ is a random variable with a given probability distribution of $$\mathbb{P}(x = a_i) = p_i$$, where $$i = 1, \ldots, n$$, and $$a_1 < \ldots < a_n$$. It is said that the probability vector of outcomes of $$p \in \mathbb{R}^n$$ belongs to the probabilistic simplex, i.e. $$P = \{ p \mid \mathbf{1}^Tp = 1, p \succeq 0 \} = \{ p \mid p_1 + \ldots + p_n = 1, p_i \ge 0 \}$$. 
    Determine if the following sets of $$p$$ are convex:
	1. $$ \alpha < \mathbb{E} f(x) < \beta$$, where $$\mathbb{E}f(x)$$ stands for expected value of $$f(x): \mathbb{R} \rightarrow \mathbb{R} $$, i.e. $$ \mathbb{E}f(x) = \sum\limits_{i=1}^n p_i f(a_i) $$ 
	1. \$$ \mathbb{E}x^2 \le \alpha $$
	1. \$$ \mathbb{V}x \le \alpha $$
1. Let $$S \subseteq \mathbb{R}^n$$ is a set of solutions to the quadratic inequality: 
    
    $$
    S = \{x \in \mathbb{R}^n \mid x^\top A x + b^\top x + c \leq 0 \}; A \in \mathbb{S}^n, b \in \mathbb{R}^n, c \in \mathbb{R}
    $$

	1. Show that if $$A \succeq 0$$, $$S$$ is convex. Is the opposite true?
	1. Show that the intersection of $$S$$ with the hyperplane defined by the $$g^\top x + h = 0, g \neq 0$$ is convex if $$A + \lambda gg^\top \succeq 0$$ for some real $$\lambda \in \mathbb{R}$$. Is the opposite true?

## Convex functions

1. Is $$f(x) = -x \ln x - (1-x) \ln (1-x)$$ convex?
1. Let $$x$$ be a real variable with the values $$a_1 < a_2 < \ldots < a_n$$ with probabilities $$\mathbb{P}(x = a_i) = p_i$$. Derive the convexity or concavity of the following functions from $$p$$ on the set of $$\left\{p \mid \sum\limits_{i=1}^n p_i = 1, p_i \ge 0 \right\}$$  
	* \$$\mathbb{E}x$$
	* \$$\mathbb{P}\{x \ge \alpha\}$$
	* \$$\mathbb{P}\{\alpha \le x \le \beta\}$$
	* \$$\sum\limits_{i=1}^n p_i \log p_i$$
	* \$$\mathbb{V}x = \mathbb{E}(x - \mathbb{E}x)^2$$
	* \$$\mathbf{quartile}(x) = {\operatorname{inf}}\left\{ \beta \mid \mathbb{P}\{x \le \beta\} \ge 0.25 \right\}$$ 
1. Show, that $$f(A) = \lambda_{max}(A)$$ - is convex, if $$A \in S^n$$
1. Prove, that $$-\log\det X$$ is convex on $$X \in S^n_{++}$$.
1. Prove, that adding $$\lambda \|x\|_2^2$$ to any convex function $$g(x)$$ ensures strong convexity of a resulting function $$f(x) = g(x) + \lambda \|x\|_2^2$$. Find the constant of the strong convexity $$\mu$$.
1. Study the following function of two variables $$f(x,y) = e^{xy}$$.
	1. Is this function convex?
	1. Prove, that this function will be convex on the line $$x = y$$.
	1. Find another set in $$\mathbb{R}^2$$, on which this function will be convex.
1.  Show, that the following function is convex on the set of all positive denominators
	
	$$
	f(x) = \dfrac{1}{x_1 - \dfrac{1}{x_2 - \dfrac{1}{x_3 - \dfrac{1}{\ldots}}}}, x \in \mathbb{R}^n
	$$

## Conjugate sets
1. Find the sets $$S^{*}, S^{**}, S^{***}$$, if 
    
    $$
    S = \{ x \in \mathbb{R}^2 \mid x_1 + x_2 \ge -1, \;\; 2x_1 - x_2 \ge 0, \;\; -x_1 + 2x_2 \ge -2\}
    $$
1. Prove, that $$K_p$$ and $$K_{p_*}$$ are inter-conjugate, i.e. $$(K_p)^* = K_{p_*}, (K_{p_*})^* = K_p$$, where $$K_p = \left\{ [x, \mu] \in \mathbb{R}^{n+1} : \|x\|_p \leq \mu \right\}, \; 1 < p < \infty$$ is the norm cone (w.r.t. $$p$$ - norm) and $$p, p_*$$ are conjugated, i.e. $$p^{-1} + p^{-1}_* = 1$$. You can assume, that $$p_* = \infty$$ if $$p = 1$$ and vice versa.
1. Suppose, $$S = S^*$$. Could the set $$S$$ be anything, but a unit ball? If it can, provide an example of another self-conjugate set. If it couldn't, prove it.
1. Find the conjugate set to the ellipsoid: 
    
    $$
     S = \left\{ x \in \mathbb{R}^n \mid \sum\limits_{i = 1}^n a_i^2 x_i^2 \le \varepsilon^2 \right\}
    $$
1. Find the conjugate cone for the exponential cone:
    
    $$
    K = \{(x, y, z) \mid y > 0, y e^{x/y} \leq z\}
    $$


## Conjugate functions
1. Find $$f^*(y)$$, if $$f(x) = px - q$$
1. Find $$f^*(y)$$, if $$f(x) =\frac{1}{2} x^T A x, \;\;\; A \in \mathbb{S}^n_{++}$$
1. Find $$f^*(y)$$, if $$f(x) = \log \left( \sum\limits_{i=1}^n e^{x_i} \right)$$
1. Prove, that if $$f(x) = g(Ax)$$, then $$f^*(y) = g^*(A^{-\top}y)$$
1. Find $$f^*(Y)$$, if $$f(X) = - \ln \det X, X \in \mathbb{S}^n_{++}$$

## Subgradient and subdifferential
1. Find $$\partial f(x)$$, if $$f(x) = \text{Leaky ReLU}(x) = \begin{cases}
    x & \text{if } x > 0, \\
    0.01x & \text{otherwise}.
\end{cases}$$
1. Find subdifferential of a function $$f(x) = \cos x$$ on the set $$X = [0, \frac32 \pi]$$.
1. Find $$\partial f(x)$$, if $$f(x) = \|Ax - b\|_1^2$$
1. Suppose, that if $$f(x) = \|x\|_\infty$$. Prove that
    $$
    \partial f(0) = \textbf{conv}\{\pm e_1, \ldots , \pm e_n\},
    $$
    where $$e_i$$ is $$i$$-th canonical basis vector (column of identity matrix).

1. Find $$\partial f(x)$$, if $$f(x) = e^{\|x\|}$$. Try do the task for an arbitrary norm. At least, try $$\|\cdot\| = \|\cdot\|_{\{2,1,\infty\}}$$.