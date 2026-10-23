function tau_c = Cdq2r(q, dq, p)
%CDQ2R 科氏/离心力矩 C(q,dq)*dq（解析闭式）
beta = p.m2 * p.L1 * p.lc2;
h = beta * sin(q(2));
tau_c = [-h*(2*dq(1)*dq(2) + dq(2)^2);
          h*dq(1)^2];
end
