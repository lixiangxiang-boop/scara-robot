function G = G2r(q, p)
%G2R 重力矩（势能梯度，支持任意平面内重力方向）
%   p.gravity = [gx, gy]（加速度矢量，默认 [0,-9.81]）
g = p.gravity(:);
q1 = q(1); q2 = q(2);
e1p  = [-sin(q1);  cos(q1)];          % d(e1)/d q1
e12p = [-sin(q1+q2); cos(q1+q2)];     % d(e12)/d q1 = d(e12)/d q2
A = p.m1*p.lc1 + p.m2*p.L1;
B = p.m2*p.lc2;
G = [-(A*(g'*e1p) + B*(g'*e12p));
     -(B*(g'*e12p))];
end
