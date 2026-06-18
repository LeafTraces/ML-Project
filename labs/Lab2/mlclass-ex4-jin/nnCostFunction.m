function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% forward
a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

Y = zeros(m, num_labels);
for i = 1:m
    Y(i, y(i)) = 1;
end

% Cost
J = (1/m) * sum(sum(-Y .* log(a3) - (1 - Y) .* log(1 - a3)));

% Add regularization to cost
J = J + (lambda/(2*m)) * (sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)));

% Backpropogation
d3 = a3 - Y;                                          
d2 = (d3 * Theta2(:, 2:end)) .* sigmoidGradient(z2);

Theta2_grad = (1/m) * (d3' * a2);
Theta1_grad = (1/m) * (d2' * a1);

% regularization
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda/m) * Theta2(:, 2:end);
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m) * Theta1(:, 2:end);

% Unroll
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end