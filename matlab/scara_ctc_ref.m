function r = scara_ctc_ref(u)
%SCARA_CTC_REF  Simulink 参考轨迹块：t -> [qd(4); dqd(4); ddqd(4)]（12×1）
%   场景与既有 SCARA 仿真一致：q0=[0;0;0.05;0] → q1=[60°;-40°;0.15;90°]，2 s 五次多项式
persistent q0 q1 T ready
if isempty(ready)
    q0 = [0; 0; 0.05; 0];
    q1 = [deg2rad(60); deg2rad(-40); 0.15; deg2rad(90)];
    T  = 2.0;
    ready = true;
end
t = u(1);
tau = min(max(t/T, 0), 1);
h = q1 - q0;
s   = 10*tau.^3 - 15*tau.^4 + 6*tau.^5;
ds  = (30*tau.^2 - 60*tau.^3 + 30*tau.^4) / T;
dds = (60*tau - 180*tau.^2 + 120*tau.^3) / T^2;
r = [q0 + h.*s; h.*ds; h.*dds];
end
