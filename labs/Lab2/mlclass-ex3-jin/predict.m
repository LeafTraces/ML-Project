function p = predict(Theta1, Theta2, X)

m = size(X, 1);

a1 = [ones(m, 1) X];
a2 = sigmoid(a1 * Theta1');

a2 = [ones(m, 1) a2];
a3 = sigmoid(a2 * Theta2');

[~, p] = max(a3, [], 2);

end