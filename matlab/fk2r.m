function xy = fk2r(q, p)
%FK2R 2 轴平面臂正运动学：xy = [x; y]（含工具偏移 p.L2 已折算在 p.L2 中）
q1 = q(1); q2 = q(2);
xy = [p.L1*cos(q1) + p.L2*cos(q1+q2);
      p.L1*sin(q1) + p.L2*sin(q1+q2)];
end
