function n2 = dist2(x, c)
% DIST2 Pairwise squared Euclidean distances (Netlab-compatible).
% n2(i,j) = ||x(i,:) - c(j,:)||^2

[ndata, dimx] = size(x);
[ncentres, dimc] = size(c);
if dimx ~= dimc
  error('Data dimension does not match dimension of centres')
end

n2 = (ones(ncentres, 1) * sum(x.^2, 2)')' + ...
     ones(ndata, 1) * sum(c.^2, 2)' - ...
     2.*(x*(c'));

% Ensure non-negative under round-off.
n2(n2 < 0) = 0;
