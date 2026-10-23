function [M, Cqd, G, F] = scara_plant_dynamics(q, qd, p, smooth)
%SCARA_PLANT_DYNAMICS  M(q)qdd + Cqd + G + F = tau
%   与既有 SCARA 模型 scara_dynamics.m 完全一致；smooth>0 时库仑摩擦用 tanh 平滑
%   （避免 sign() 在零速附近的数值极限环，仿真里必须这么做）。
if nargin < 3, p = scara_plant_params(); end
if nargin < 4, smooth = 0; end
q = q(:); qd = qd(:);
c2 = cos(q(2)); s2 = sin(q(2));
M = [ p.pi1 + p.pi2 + 2*p.pi3*c2 + p.I4,  p.pi2 + p.pi3*c2 + p.I4,  0,     p.I4;
      p.pi2 + p.pi3*c2 + p.I4,            p.pi2 + p.I4,              0,     p.I4;
      0,                                   0,                         p.piZ, 0;
      p.I4,                                p.I4,                      0,     p.pi4];
dM = -s2*[2*p.pi3, p.pi3, 0, 0;
          p.pi3,   0,     0, 0;
          0,       0,     0, 0;
          0,       0,     0, 0];
C = zeros(4);
for i = 1:4
    for j = 1:4
        s = 0;
        for k = 1:4
            a = 0; if k == 2, a = dM(i,j); end
            b = 0; if j == 2, b = dM(i,k); end
            c = 0; if i == 2, c = dM(k,j); end
            s = s + (a + b - c)*qd(k);
        end
        C(i,j) = 0.5*s;
    end
end
Cqd = C*qd;
G = [0; 0; -p.mt*p.g; 0];
if smooth > 0
    F = p.Fv(:).*qd + p.Fc(:).*tanh(qd/smooth);
else
    F = p.Fv(:).*qd + p.Fc(:).*sign(qd);
end
end
