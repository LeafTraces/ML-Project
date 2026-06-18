function p = predictOneVsAll(all_theta, X)

m = size(X, 1);

X = [ones(m, 1) X];

probs = sigmoid(X * all_theta');

[~, p] = max(probs, [], 2);

end