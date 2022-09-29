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

## Convexity

