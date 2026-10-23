function qdd = scara_plant_block(u)
%SCARA_PLANT_BLOCK  Simulink 被控对象块：u = [q(4); dq(4); tau(4)] → qdd(4)
persistent p ready
if isempty(ready)
    p = scara_plant_params();
    ready = true;
end
q  = u(1:4);
dq = u(5:8);
tau = u(9:12);
[M, Cqd, G, F] = scara_plant_dynamics(q, dq, p, 0.02);   % 0.02：库仑摩擦平滑宽度
qdd = M \ (tau - Cqd - G - F);
end
