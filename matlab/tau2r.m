function tau = tau2r(q, dq, ddq, p)
%TAU2R 逆动力学：tau = M(q)*ddq + C(q,dq)*dq + G(q)
tau = M2r(q, p)*ddq(:) + Cdq2r(q, dq, p) + G2r(q, p);
end
