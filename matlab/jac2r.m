function J = jac2r(q, p)
%JAC2R 2 轴平面臂几何雅可比（世界系，2x2）
q1 = q(1); q2 = q(2);
s1 = sin(q1); c1 = cos(q1); s12 = sin(q1+q2); c12 = cos(q1+q2);
J = [-p.L1*s1 - p.L2*s12, -p.L2*s12;
      p.L1*c1 + p.L2*c12,  p.L2*c12];
end
