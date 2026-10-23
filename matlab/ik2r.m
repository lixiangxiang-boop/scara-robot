function q = ik2r(xy, p, elbow)
%IK2R 解析逆运动学（余弦定理）
%   elbow: 'down'（q2<0，默认）或 'up'（q2>0）
if nargin < 3, elbow = 'down'; end
r2 = xy(1)^2 + xy(2)^2;
c2 = (r2 - p.L1^2 - p.L2^2) / (2*p.L1*p.L2);
c2 = max(min(c2, 1), -1);
q2 = acos(c2);
if strcmp(elbow, 'down'), q2 = -q2; end
q1 = atan2(xy(2), xy(1)) - atan2(p.L2*sin(q2), p.L1 + p.L2*cos(q2));
q = [q1; q2];
end
