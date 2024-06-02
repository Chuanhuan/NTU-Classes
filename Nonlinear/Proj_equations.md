 # ELBO derivative wrt x and $\theta$

$$\begin{align*}\nabla_{\theta}\mathrm{ELBO}(q) &=\nabla_{\theta}\mathbb{E}_{q}{\big[}\log p(\mathbf{x},\mathbf{z})-\log q_{\theta}(\mathbf{z}){\big]} \\
 &=\mathbb{E}_{q}{\big[}\nabla_{\theta}\log q_{\theta}(\mathbf{z}){\big(}\log p(\mathbf{x},\mathbf{z})-\log q_{\theta}(\mathbf{z}){\big)}{\big]}\end{align*}$$

 $$\begin{align*} &=\mathbb{E}_{q}{\big[}\nabla_{\theta}\log x p_{\theta}(\mathbf{z}){\big(}\log p_{\theta}-\log x p_{\theta}(\mathbf{z}){\big)}{\big]}  \\ 
&= \mathbb{E}_{q}{\big[}\nabla_{\theta}\log x p_{\theta}(\mathbf{z}){\big(} \log \frac { p_{\theta}}  {x p_{\theta}(\mathbf{z})}  {\big)}{\big]} \\
&= \mathbb{E}_{q}{\big[}\nabla_{\theta}\log x p_{\theta}(\mathbf{z}){\big(} \log \frac { p_{\theta}}  {x p_{\theta}(\mathbf{z})}  {\big)}{\big] }\\
&= \mathbb{E}_{q}{\big[}\nabla_{\theta}\log x p_{\theta}(\mathbf{z}){\big(} -\log x {\big)}{\big] }\\
&= \int p_{\theta} x {\big[}-\frac{1}{ p_{\theta}}{\big] }dx \\
&= -\frac{x^2}{ 2} +c
\end{align*} $$



$$\begin{align*}\nabla_{x}\mathrm{ELBO}(q) &=\nabla_{x}\mathbb{E}_{q}{\big[}\log p(\mathbf{x},\mathbf{z})-\log q_{\theta}(\mathbf{z}){\big]} \\
&= \int x p(\mathbf{x},\mathbf{z}) \log\frac{1}{ x}dx \\
&= -x p(\mathbf{x},\mathbf{z})\log x +c
 \end{align*}$$