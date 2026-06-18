function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y);

h = sigmoid(X * theta);
J = (1/m) * sum(-y .* log(h) - (1 - y) .* log(1-h)) + (lambda/(2*m)) * sum(theta(2:end).^2);
grad = (1/m) * (X' * (h-y));
grad(2:end) = grad(2:end) + (lambda / m) * theta(2:end);

grad = grad(:);

end