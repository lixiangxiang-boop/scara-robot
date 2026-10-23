function M = M2r(q, p)
%M2R 质量矩阵（解析闭式）
%   M11 = I1+I2+m1*lc1^2+m2*(L1^2+lc2^2+2*L1*lc2*cos q2)
%   M12 = I2+m2*(lc2^2+L1*lc2*cos q2)，M22 = I2+m2*lc2^2
c2 = cos(q(2));
M11 = p.I1 + p.I2 + p.m1*p.lc1^2 + p.m2*(p.L1^2 + p.lc2^2 + 2*p.L1*p.lc2*c2);
M12 = p.I2 + p.m2*(p.lc2^2 + p.L1*p.lc2*c2);
M22 = p.I2 + p.m2*p.lc2^2;
M = [M11, M12; M12, M22];
end
