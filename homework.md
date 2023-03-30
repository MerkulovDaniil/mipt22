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

## General optimization problem

1. Consider the problem of projection some point $$y \in \mathbb{R}^n,  y \notin \Delta^n$$ onto the unit simplex $$\Delta^n$$. Find 2 ways to solve the problem numerically and compare them in terms of the total computational time, memory requirements and iteration number for $$n = 10, 100, 1000$$. 

	$$
	\begin{split}
	& \|x - y \|_2^2 \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & 1^\top x = 1, \\
	& x \succeq 0 
	\end{split}
	$$

1. Give an explicit solution of the following LP.
	
	$$
	\begin{split}
	& c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & Ax = b
	\end{split}
	$$

1. Give an explicit solution of the following LP.
	
	$$
	\begin{split}
	& c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & 1^\top x = 1, \\
	& x \succeq 0 
	\end{split}
	$$

	This problem can be considered as a simplest portfolio optimization problem.

1. Give an explicit solution of the following LP.
	
	$$
	\begin{split}
	& c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & 1^\top x = \alpha, \\
	& 0 \preceq x \preceq 1,
	\end{split}
	$$

	where $$\alpha$$ is an integer between $$0$$ and $$n$$. What happens if $$\alpha$$ is not an integer (but satisfies $$0 \leq \alpha \leq n$$)? What if we change the equality to an inequality $$1^\top x \leq \alpha$$?

1. Give an explicit solution of the following QP.
	
	$$
	\begin{split}
	& c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & x^\top A x \leq 1,
	\end{split}
	$$

	where $$A \in \mathbb{S}^n_{++}, c \neq 0$$. What is the solution if the problem is not convex $$(A \notin \mathbb{S}^n_{++})$$ (Hint: consider eigendecomposition of the matrix: $$A = Q \mathbf{diag}(\lambda)Q^\top = \sum\limits_{i=1}^n \lambda_i q_i q_i^\top$$ and different cases of $$\lambda >0, \lambda=0, \lambda<0$$)?

1. Give an explicit solution of the following QP.
	
	$$
	\begin{split}
	& c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & (x - x_c)^\top A (x - x_c) \leq 1,
	\end{split}
	$$

	where $$A \in \mathbb{S}^n_{++}, c \neq 0, x_c \in \mathbb{R}^n$$.

1. Give an explicit solution of the following QP.
	
	$$
	\begin{split}
	& x^\top Bx \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & x^\top A x \leq 1,
	\end{split}
	$$

	where $$A \in \mathbb{S}^n_{++}, B \in \mathbb{S}^n_{+}$$.

1.  Consider the equality constrained least-squares problem
	
	$$
	\begin{split}
	& \|Ax - b\|_2^2 \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & Cx = d,
	\end{split}
	$$

	where $$A \in \mathbb{R}^{m \times n}$$ with $$\mathbf{rank }A = n$$, and $$C \in \mathbb{R}^{k \times n}$$ with $$\mathbf{rank }C = k$$. Give the KKT conditions, and derive expressions for the primal solution $$x^*$$ and the dual solution $$\lambda^*$$.

1. Derive the KKT conditions for the problem
	
	$$
	\begin{split}
	& \mathbf{tr \;}X - \log\text{det }X \to \min\limits_{X \in \mathbb{S}^n_{++} }\\
	\text{s.t. } & Xs = y,
	\end{split}
	$$

	where $$y \in \mathbb{R}^n$$ and $$s \in \mathbb{R}^n$$ are given with $$y^\top s = 1$$. Verify that the optimal solution is given by

	$$
	X^* = I + yy^\top - \dfrac{1}{s^\top s}ss^\top
	$$

1.  **Supporting hyperplane interpretation of KKT conditions**. Consider a **convex** problem with no equality constraints
	
	$$
	\begin{split}
	& f_0(x) \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & f_i(x) \leq 0, \quad i = [1,m]
	\end{split}
	$$

	Assume, that $$\exists x^* \in \mathbb{R}^n, \mu^* \in \mathbb{R}^m$$ satisfy the KKT conditions
	
	$$
	\begin{split}
    & \nabla_x L (x^*, \mu^*) = \nabla f_0(x^*) + \sum\limits_{i=1}^m\mu_i^*\nabla f_i(x^*) = 0 \\
    & \mu^*_i \geq 0, \quad i = [1,m] \\
    & \mu^*_i f_i(x^*) = 0, \quad i = [1,m]\\
    & f_i(x^*) \leq 0, \quad i = [1,m]
	\end{split}
	$$

	Show that

	$$
	\nabla f_0(x^*)^\top (x - x^*) \geq 0
	$$

	for all feasible $$x$$. In other words the KKT conditions imply the simple optimality criterion or $$\nabla f_0(x^*)$$ defines a supporting hyperplane to the feasible set at $$x^*$$.

## Duality

1.  **Fenchel + Lagrange = ♥.** Express the dual problem of
	
	$$
	\begin{split}
	& c^\top x\to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & f(x) \leq 0
	\end{split}
	$$

	with $$c \neq 0$$, in terms of the conjugate function $$f^*$$. Explain why the problem you give is convex. We do not assume $$f$$ is convex.

1. **Minimum volume covering ellipsoid.** Let we have the primal problem:
	
	$$
	\begin{split}
	& \ln \text{det} X^{-1} \to \min\limits_{X \in \mathbb{S}^{n}_{++} }\\
	\text{s.t. } & a_i^\top X a_i \leq 1 , i = 1, \ldots, m
	\end{split}
	$$

	1. Find Lagrangian of the primal problem
	1. Find the dual function
	1. Write down the dual problem
	1. Check whether problem holds strong duality or not
	1. Write down the solution of the dual problem
	
1. **A penalty method for equality constraints.** We consider the problem of minimization
	$$
	\begin{split}
	& f_0(x) \to \min\limits_{x \in \mathbb{R}^{n} }\\
	\text{s.t. } & Ax = b,
	\end{split}
	$$
	
	where $$f_0(x): \mathbb{R}^n \to\mathbb{R} $$ is convex and differentiable, and $$A \in \mathbb{R}^{m \times n}$$ with $$\mathbf{rank }A = m$$. In a quadratic penalty method, we form an auxiliary function

	$$
	\phi(x) = f_0(x) + \alpha \|Ax - b\|_2^2,
	$$
	
	where $$\alpha > 0$$ is a parameter. This auxiliary function consists of the objective plus the penalty term $$\alpha \|Ax - b\|_2^2$$. The idea is that a minimizer of the auxiliary function, $$\tilde{x}$$, should be an approximate solution of the original problem. Intuition suggests that the larger the penalty weight $$\alpha$$, the better the approximation $$\tilde{x}$$ to a solution of the original problem. Suppose $$\tilde{x}$$ is a minimizer of $$\phi(x)$$. Show how to find, from $$\tilde{x}$$, a dual feasible point for the original problem. Find the corresponding lower bound on the optimal value of the original problem.
	
1. **Analytic centering.** Derive a dual problem for
	
	$$
	-\sum_{i=1}^m \log (b_i - a_i^\top x) \to \min\limits_{x \in \mathbb{R}^{n} }
	$$

	with domain $$\{x \mid a^\top_i x < b_i , i = [1,m]\}$$. 
	
	First introduce new variables $$y_i$$ and equality constraints $$y_i = b_i − a^\top_i x$$. (The solution of this problem is called the analytic center of the linear inequalities $$a^\top_i x \leq b_i ,i = [1,m]$$.  Analytic centers have geometric applications, and play an important role in barrier methods.) 
	
## Linear Programming

1. **📱🎧💻 Covers manufacturing.** Random Corp is producing covers for following products: 
	* 📱 phones
	* 🎧 headphones
	* 💻 laptops

	The company’s production facilities are such that if we devote the entire production to headphones covers, we can produce 5000 of them in one day. If we devote the entire production to phone covers or laptop covers, we can produce 4000 or 2000 of them in one day. 

	The production schedule is one week (6 working days), and the week’s production must be stored before distribution. Storing 1000 headphones covers (packaging included) takes up 30 cubic feet of space. Storing 1000 phone covers (packaging included) takes up 50 cubic feet of space, and storing 1000 laptop covers (packaging included) takes up 220 cubic feet of space. The total storage space available is 1500 cubic feet. 
	

	Due to commercial agreements with Random Corp has to deliver at least 4500 headphones covers and 3000 laptop covers per week in order to strengthen the product’s diffusion. 

	The marketing department estimates that the weekly demand for headphones covers, phone, and laptop covers does not exceed 9000 and 14000, and 7000 units, therefore the company does not want to produce more than these amounts for headphones, phone, and laptop covers. 

	Finally, the net profit per each headphones cover, phone cover, and laptop cover is \$5, \$7, and \$12, respectively.

	The aim is to determine a weekly production schedule that maximizes the total net profit.

	1. Write a Linear Programming formulation for the problem.	Use following variables:

		* $$y_1$$ = number of headphones covers produced over the week,  
		* $$y_2$$ = number of phone covers produced over the week,  
		* $$y_3$$ = number of laptop covers produced over the week. 

	1. Find the solution to the problem using [PyOMO](http://www.pyomo.org)
		```python
		!pip install pyomo
		! sudo apt-get install glpk-utils --quiet  # GLPK
		! sudo apt-get install coinor-cbc --quiet  # CoinOR
		```

1. Prove the optimality of the solution
	
	$$
	x^\top = \left(\frac{5}{26} , \frac{5}{2}, \frac{27}{26}\right)
	$$
	
	to the following linear programming problem:
	
	$$
	\begin{split}
	& 9x_1 + 14x_2 + 7x_3 \to \max\limits_{x \in \mathbb{R}^3 }\\
	\text{s.t. } & 2x_1 + x_2 + 3x_3 \leq 6 \\
	& 5x_1 + 4x_2 + x_3 \leq 12 \\
	& 2x_2 \leq 5,
	\end{split}
	$$

	but you cannot use any numerical algorithm here.

1. Transform the following linear program into an equivalent linear program in standard form $$\left(c^\top x \to \max\limits_{x\in \mathbb{R}^n} : Ax = b,x ≥ 0\right)$$:

	$$
	\begin{split}
	& x_1−x_2 \to \min\limits_{x \in \mathbb{R}^2 }\\
	\text{s.t. } & 2x_1 + x_2 \geq 3 \\
	& 3x_1 − x_2 \leq 7 \\
	& x_1 \geq 0
	\end{split}
	$$

1. Consider:

	$$
	\begin{split}
	& 4x_1 + 5x_2 + 2x_3 \to \max\limits_{x \in \mathbb{R}^3 }\\
	\text{s.t. } & 2x_1 - x_2 + 2x_3 \leq 9 \\
	& 3x_1 + 5x_2 + 4x_3 \leq 8 \\
	& x_1 + x_2 + 2x_3 \leq 2 \\
	& x_1, x_2, x_3 \geq 0,
	\end{split}
	$$

	1. Find an optimal solution to the Linear Programming problem using the simplex method.
	1. Write the dual linear program. Find an optimal dual solution. Do we have strong duality here?

# Домашнее задание 3

The file should be sent in the `.pdf` format created via $\LaTeX$ or [typora](<https://typora.io/>) or printed from pdf with the colab\jupyter notebook. The only handwritten part, that could be included in the solution are the figures and illustrations.

**Deadline: 09.04.23 21:59:59**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg#button)](https://colab.research.google.com/github/MerkulovDaniil/mipt22/blob/main/notebooks/23_HW_1.ipynb)
